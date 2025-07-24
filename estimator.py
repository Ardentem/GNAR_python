import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

class GNAR_estimator:
    def __init__(self, Y, CV, network, seed=42, G=8):
        """ Initialize the GNAR estimator with parameters   
        Y: N*T matrix of observations   
        CV: N*p matrix of covariates   
        network: N*N*T matrix of network connections      
        seed: Random seed for reproducibility   
        G: Number of groups     
        """
        self.Y = Y
        self.CV = CV
        self.network = network
        self.N, self.T = Y.shape
        self.p = CV.shape[1]
        self.G = G
        self.seed = seed
        np.random.seed(seed)
        self.beta = None
        self.group = None

    def ridgeOLS(self, X, y):
        """ Ridge Ordinary Least Squares estimation """
        # 增加ridge tuning parameter
        XTX = X @ X.T
        # 计算lambda值并添加到对角线
        lambda_values = []
        for j in range(X.shape[0]):
            x_it_norm_squared = np.sum(X[j, :] ** 2)
            lambda_j = 0.01 * x_it_norm_squared / (X.shape[1] + 1) + 1e-6
            lambda_values.append(lambda_j)
        # 在XTX对角线上加入lambda值
        XTX += np.diag(lambda_values)
        beta = np.linalg.inv(XTX) @ X @ y
        return beta

    def initialize_base_OLS(self,time_varying=False):
        """ Initialize for kmean
        time_varying: bool, whether the model is time-varying
        return:
        beta: N*N matrix of coefficients
        v: N vector of momentum coefficients
        if not time_varying:
            fi: N vector of intercepts
        else:
            gamma: N*p matrix of CV coefficients
        """
        if not time_varying:
            # 删去第0天并对每个个体去均值
            Y_tire = self.Y[:, 1:] - np.mean(self.Y[:, 1:], axis=1, keepdims=True)
            # 删去最后一天并对每个个体去均值
            Y_tire_lag = self.Y[:, :-1] - np.mean(self.Y[:, :-1], axis=1, keepdims=True)
            # 初始化beta N+1* N
            beta = np.zeros((self.N + 1, self.N))
            fi = np.zeros(self.N)
            for i in range(self.N):
                Y_tire_lag_i = Y_tire_lag[i, :].reshape(1, -1) # 1*T-1
                Y_tire_i = Y_tire[i, :] # 1*T-1
                # 取Network
                WY_tire_lag_i = self.network[i, :, :-1] * Y_tire_lag   # N*T-1
                # 拼接
                X = np.concatenate([WY_tire_lag_i,Y_tire_lag_i], axis=0) # N+1*T-1
                # 计算beta
                beta[:, i] = self.ridgeOLS(X, Y_tire_i)  # T-1*1
                # 计算所有时间的网络均值
                net_mean = np.mean(self.network[i, :, :-1], axis=1)  # N*1
                fi[i] = np.mean(self.Y[i,1:]) - np.sum(beta[:-1, i] * net_mean * np.mean(self.Y[:, :-1], axis=1))- beta[-1, i] * np.mean(self.Y[i, :-1])
            return beta[:-1,:],beta[-1,:], fi
        else:
            # 初始化beta N+1+p* N*T
            beta = np.zeros((self.N + 1 + self.p, self.N))
            for i in range(self.N):
                # 取Network
                Y_lag = self.Y[:, :-1]
                WY_lag_i = self.network[i, :, :-1] * Y_lag
                Y_lag_i = Y_lag[i, :].reshape(1, -1)
                X = np.concatenate([WY_lag_i, Y_lag_i, self.CV[i, :, :-1]], axis=0)
                beta[:, i] = self.ridgeOLS(X, self.Y[i, 1:])  # T-1*1
            return beta[:self.N,:],beta[self.N,:], beta[self.N+1:,:]

    def k_means_clustering(self, method='networkeffect', time_varying=False):
        """ Perform k-means clustering based on the specified method
        method = networkeffect: 'momentum', 'fixedeffect', or 'networkeffect'
        !!! Note: 'complete' a test method for k-means clustering
        !!! networkeffect is recommended for better results
        """
        beta, v , fi = self.initialize_base_OLS(time_varying=time_varying)
        if method == 'momentum':
            # 使用动量系数进行k-means聚类
            Kres = KMeans(n_clusters=self.G, random_state=self.seed).fit(v.reshape(-1, 1))
            self.group = Kres.labels_
        elif method == 'fixedeffect':
            # 使用固定效应进行k-means聚类
            if not time_varying:
                Kres = KMeans(n_clusters=self.G, random_state=self.seed).fit(fi.reshape(-1, 1))
            else:
                Kres = KMeans(n_clusters=self.G, random_state=self.seed).fit(fi.T)
            self.group = Kres.labels_
        elif method == 'networkeffect' or method == 'complete':
            # 使用网络效应进行k-means聚类, Two steps
            # 先对beta进行kmeans G^2个组
            Kres1 = KMeans(n_clusters=self.G**2, random_state=self.seed).fit(beta.reshape(-1, 1))
            c = Kres1.labels_.reshape(self.N, self.N)
            beta_net = np.zeros((self.N,1+self.G**2))
            for l in range(self.G**2):
                beta_ = beta.copy()
                # 只保留c=1的beta，其他设为0
                beta_[c != l] = 0
                # 对每个组的beta求均值
                beta_net[:,1+l] = np.mean(beta_, axis=0) / np.sum(c == l, axis=0)
            beta_net[:, 0] = v
            # fillna
            beta_net = np.nan_to_num(beta_net, nan=0.0)
            if method == 'complete':
                beta_net = np.concatenate([beta_net, fi.T], axis=1)
            Kres2 = KMeans(n_clusters=self.G, random_state=self.seed).fit(beta_net)
            self.group = Kres2.labels_
        return self.group

    def cul_para_OLS(self):
        """ Calculate the parameters using self.group
        return:
        beta_all: (G+p+1)*G matrix of coefficients
        """
        X = np.zeros((self.N, self.G + self.p + 1, self.T - 1))
        Y = np.zeros((self.N, self.T - 1))
        beta_all = np.zeros((self.G + self.p + 1, self.G))
        for i in range(self.N):
            # 取Network 第i列
            X_i = np.zeros((self.G+self.p+1, self.T-1))
            Y_i = self.Y[i, 1:]
            for g in range(self.G):
                WY = self.network[i, :, :-1] * self.Y[:, :-1] * np.repeat((self.group == g).reshape(-1, 1), self.T-1, axis=1)  # N*T-1
                X_i[g, :] = np.sum(WY, axis=0)
            X_i[self.G, :] = self.Y[i, :-1]
            X_i[self.G+1:self.G+self.p+1, :] = self.CV[i, :, :-1]
            X[i, :, :] = X_i
            Y[i, :] = Y_i
        for g in range(self.G):
            if np.sum(self.group == g) == 0:
                # 初始化系数的时候不会出现某个组为0的情况
                beta_all[:, g] = self.beta[:, g]
                continue
            Xg = np.concatenate(X[self.group == g],axis=1)  # N_obs*(G+p+1)*(T-1)
            yg = np.concatenate(Y[self.group == g])  # N_obs*(T-1)
            # 计算OLS系数
            beta = np.linalg.inv(Xg @ Xg.T) @ Xg @ yg
            beta_all[:, g] = beta
        return beta_all

    def loss_function(self, beta=None, group=None):
        """ Calculate the loss function based on the parameters beta
        beta: (G+p+1)*G matrix of coefficients, if None, use self.beta
        group: N vector of group labels, if None, use self.group
        return:
        loss: float, the loss value
        """
        if beta is None:
            beta = self.beta
        if group is None:
            group = self.group
        #按照N个节点对应的组，构建N*N的系数矩阵，其中i，j节点的系数为beta[group[j], group[i]]
        self.coef = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.coef[i, j] = beta[group[j], group[i]]
        # 按照N个节点对应的组，把G个动量系数v构建长度为N的动量系数
        self.v_coef = np.zeros(self.N)
        for i in range(self.N):
            self.v_coef[i] = beta[self.G, group[i]]
        # 按照N个节点对应的组，把G*p个CV系数gamma构建N*p的CV系数
        self.gamma_coef = np.zeros((self.N, self.p))
        for i in range(self.N):
            self.gamma_coef[i, :] = beta[self.G+1:self.G+self.p+1, group[i]]
        # 将N*N*T的network矩阵与系数矩阵相乘，得到N*N*T的网络系数矩阵
        self.network_coef = np.zeros((self.N, self.N, self.T))
        for t in range(self.T):
            self.network_coef[:, :, t] = self.network[:, :, t] * self.coef
        #计算每一天的 Yt = v@Yt-1 + network_coef@Yt-1 + gamma@CV + eps，注意是矩阵乘法
        Y_est = np.zeros((self.N, self.T-1))
        for t in range(1, self.T):
            # 计算上一天的Yt-1
            Y_prev = self.Y[:, t-1]
            # 计算动量部分
            momentum = self.v_coef * Y_prev
            # 计算网络部分
            network_part = self.network_coef[:, :, t-1] @ Y_prev
            # 计算CV部分（先乘再求和）
            CV_part = np.sum(self.gamma_coef * self.CV[:, :, t-1], axis=1)
            # 计算当天的Yt
            Y_est[:, t-1] = momentum + network_part + CV_part
        loss = np.sum((Y_est - self.Y[:, 1:]) ** 2) / (self.N * (self.T-1))
        return loss

    def update_para_group(self):
        """ Update the group labels based on the loss function
        return:
        update_count: int, the number of nodes whose group labels are updated
        """
        update_count = 0
        if self.group is None:
            raise ValueError("Group is not initialized, Run k_means_clustering() first")
        # update beta
        self.beta = self.cul_para_OLS()
        for i in tqdm(range(self.N)):
            i_group_loss = []
            for g in range(self.G):
                group = self.group.copy()
                group[i] = g
                loss = self.loss_function(group=group)
                i_group_loss.append(loss)
            # 找到最小损失对应的组
            if self.group[i] != np.argmin(i_group_loss):
                update_count += 1
                self.group[i] = np.argmin(i_group_loss)
        return update_count
    
    def update_all_para(self, max_iter=100):
        """ Update all parameters until convergence or max_iter reached
        return:
        group: N vector of group labels
        beta: (G+p+1)*G matrix of coefficients
        """
        for epoch in range(max_iter):
            update_count = self.update_para_group()
            print(f"Epoch {epoch}: {update_count} nodes updated")
            if update_count == 0:
                print("Convergence reached")
                break
        return self.group, self.beta