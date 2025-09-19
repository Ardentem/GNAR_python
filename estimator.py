import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy import stats

class GNAR_estimator:
    def __init__(self, Y, CV, network, seed=42, G=5, X=None):
        """ Initialize the GNAR estimator with parameters
        Y: N*T matrix of observations
        CV: N*p matrix of covariates
        network: N*N*T matrix of network connections
        seed: Random seed for reproducibility
        G: Number of groups     
        N: Number of nodes
        X: N*T matrix of variables (if autoregression is not used)
        """
        self.Y = Y
        self.X = X
        self.CV = CV
        self.network = network
        self.N, self.T = Y.shape
        self.p = CV.shape[1]
        self.G = G
        self.seed = seed
        np.random.seed(seed)
        self.beta = None
        self.group = None
        self.group0 = None # 初始分组
    
    def ridgeOLS(self, X, y):
        """ Ridge Ordinary Least Squares estimation (vectorized & faster) """
        '''
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
        '''
        XTX = X @ X.T
        # 计算每一行的平方和并生成 lambda 向量（向量化）
        x_norm_sq = np.sum(X**2, axis=1)
        lambda_values = 0.01 * x_norm_sq / (X.shape[1] + 1) + 1e-6
        # 在 XTX 对角线上加入 lambda
        XTX += np.diag(lambda_values)
        # 使用 np.linalg.solve 替代 inv，提高速度和稳定性
        beta = np.linalg.solve(XTX, X @ y)
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
        if self.X is None:
            if not time_varying:
                # 删去第0天并对每个个体去均值
                Y_tire = self.Y[:, 1:] - np.mean(self.Y[:, 1:], axis=1, keepdims=True)
                # 删去最后一天并对每个个体去均值
                Y_tire_lag = self.Y[:, :-1] - np.mean(self.Y[:, :-1], axis=1, keepdims=True)
                # 初始化beta N+1* N
                beta = np.zeros((self.N + 1, self.N))
                fi = np.zeros(self.N)
                Y_mean_all = np.mean(self.Y[:, :-1], axis=1)  # N*T-1 均值
                Y_i_mean_all = np.mean(self.Y, axis=1)  # N*1
                for i in range(self.N):
                    # 构造 X
                    WY_tire_lag_i = self.network[i, :, :-1] * Y_tire_lag  # N*T-1
                    X = np.vstack([WY_tire_lag_i, Y_tire_lag[i, :].reshape(1, -1)])  # (N+1)*T-1
                    # 计算 beta
                    beta[:, i] = self.ridgeOLS(X, Y_tire[i, :])
                    # 网络均值
                    net_mean = np.mean(self.network[i, :, :-1], axis=1)  # N*1
                    # 计算 fi[i]，向量化内部乘法
                    fi[i] = (Y_i_mean_all[i, 1:] if Y_i_mean_all.ndim>1 else Y_i_mean_all[i]) \
                            - (beta[:-1, i] * net_mean * Y_mean_all).sum() \
                            - beta[-1, i] * np.mean(self.Y[i, :-1])
                '''
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
                '''
                return beta[:-1,:],beta[-1,:], fi
            else:
                # 初始化beta N+1+p* N*T
                beta = np.zeros((self.N + 1 + self.p, self.N))
                # 预处理Y_lag，去掉最后一列
                Y_lag = self.Y[:, :-1]  # shape (N, T-1)
                # 处理每个节点
                for i in range(self.N):
                    # WY_lag_i
                    WY_lag_i = self.network[i, :, :-1] * Y_lag  # shape (N, T-1)
                    # Y_lag_i
                    Y_lag_i = Y_lag[i, :].reshape(1, -1)        # shape (1, T-1)
                    # CV
                    CV_i = self.CV[i, :, :-1]                   # shape (p, T-1)
                    # 使用 np.vstack 替代 np.concatenate
                    X = np.vstack([WY_lag_i, Y_lag_i, CV_i])   # shape (N+1+p, T-1)
                    # 调用ridgeOLS
                    beta[:, i] = self.ridgeOLS(X, self.Y[i, 1:])
                '''
                for i in range(self.N):
                    Y_lag = self.Y[:, :-1]
                    WY_lag_i = self.network[i, :, :-1] * Y_lag
                    Y_lag_i = Y_lag[i, :].reshape(1, -1)
                    X = np.concatenate([WY_lag_i, Y_lag_i, self.CV[i, :, :-1]], axis=0)
                    beta[:, i] = self.ridgeOLS(X, self.Y[i, 1:])  # T-1*1
                '''
                return beta[:self.N,:],beta[self.N,:], beta[self.N+1:,:]
        else:
            # 如果使用自回归，则X为N*T矩阵
            if not time_varying:
                # 去均值
                X_tire = self.X - np.mean(self.X, axis=1, keepdims=True)
                # 初始化beta N+1* N
                beta = np.zeros((self.N + 1, self.N))
                fi = np.zeros(self.N)
                # 预计算Y的均值
                Y_mean = np.mean(self.Y, axis=1)
                for i in range(self.N):
                    X_tire_i = X_tire[i:i+1, :]  # 直接切片，不用reshape
                    # 取Network并乘以X_tire
                    WX_tire_i = self.network[i, :, :] * X_tire  # N*T
                    # 拼接
                    X = np.vstack([WX_tire_i, X_tire_i])  # N+1*T
                    # 计算beta
                    beta[:, i] = self.ridgeOLS(X, self.Y[i, :])
                    # 网络均值
                    net_mean = np.mean(self.network[i, :, :], axis=1)  # N
                    # 计算fi
                    fi[i] = Y_mean[i] - np.sum(beta[:-1, i] * net_mean * Y_mean) - beta[-1, i] * Y_mean[i]
                '''
                for i in range(self.N):
                    X_tire_i = X_tire[i, :].reshape(1, -1)
                    # 取Network
                    WX_tire_i = self.network[i, :, :] * X_tire  # N*T
                    # 拼接
                    X = np.concatenate([WX_tire_i, X_tire_i], axis=0)  # N+1*T
                    # 计算beta
                    beta[:, i] = self.ridgeOLS(X, self.Y[i, :])
                    # 计算所有时间的网络均值
                    net_mean = np.mean(self.network[i, :, :], axis=1)  # N*1
                    fi[i] = np.mean(self.Y[i, :]) - np.sum(beta[:-1, i] * net_mean * np.mean(self.Y, axis=1)) - beta[-1, i] * np.mean(self.Y[i, :])
                '''
                return beta[:-1,:], beta[-1,:], fi
            else:
                beta = np.zeros((self.N + 1 + self.p, self.N))
                # 预处理X，避免循环里重复reshape
                X_all = self.X
                CV_all = self.CV
                for i in range(self.N):
                    # X_i直接切片，不用reshape
                    X_i = X_all[i:i+1, :]  # shape 1*T
                    # WX_i
                    WX_i = self.network[i, :, :] * X_all  # N*T
                    # CV_i
                    CV_i = CV_all[i, :, :]  # p*T
                    # 用vstack替代concatenate
                    X = np.vstack([WX_i, X_i, CV_i])  # (N+1+p)*T
                    # 计算beta
                    beta[:, i] = self.ridgeOLS(X, self.Y[i, :])
                '''
                # 初始化beta N+1+p* N*T
                beta = np.zeros((self.N + 1 + self.p, self.N))
                for i in range(self.N):
                    X_i = self.X[i, :].reshape(1, -1)
                    WX_i = self.network[i, :, :] * self.X  # N*T
                    X = np.concatenate([WX_i, X_i, self.CV[i, :, :]], axis=0)
                    beta[:, i] = self.ridgeOLS(X, self.Y[i, :])
                '''
                return beta[:self.N,:], beta[self.N,:], beta[self.N+1:,:]

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
    
    def cul_OLS_result(self, Xg, yg, beta):
        # 预测和残差
        y_hat = Xg.T @ beta
        residuals = yg - y_hat
        # 计算稳健协方差矩阵（White标准误，Sandwich估计）
        # S = sum_i (e_i^2 * x_i @ x_i.T)
        '''
        S = np.zeros((Xg.shape[0], Xg.shape[0]))
        for i in range(Xg.shape[1]):
            xi = Xg[:, i][:, np.newaxis]  # (k, 1)
            S += residuals[i]**2 * (xi @ xi.T)  # (k, k)
        '''
        R = residuals**2  # shape (n,)
        X_weighted = Xg * R[np.newaxis, :]  # shape (k, n)
        # 矩阵乘法
        S = X_weighted @ Xg.T  # shape (k, k)
        XGXG_inv = np.linalg.inv(Xg @ Xg.T)
        robust_var = XGXG_inv @ S @ XGXG_inv  # (k, k)
        # 标准误
        robust_se = np.sqrt(np.diag(robust_var))
        # t统计量
        t_stats = beta.flatten() / robust_se
        # p值（假设自由度 n - k）
        df = Xg.shape[1] - Xg.shape[0]
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
        yg = yg.flatten()
        y_hat = y_hat.flatten()
        # 总离差平方和（TSS）
        TSS = np.sum((yg - np.mean(yg)) ** 2)
        # 残差平方和（RSS）
        RSS = np.sum((yg - y_hat) ** 2)
        # 决定系数 R^2
        R2 = 1 - RSS / TSS
        n = yg.shape[0]
        k = Xg.shape[0]
        R2_adj = 1 - (1 - R2) * (n - 1) / (n - k)
        # 返回结果
        return robust_se, t_stats, p_values, R2, R2_adj, n

    def cul_para_OLS(self, final=False):
        """ Calculate the parameters using self.group
        final: bool, whether to return the final parameters
        If final is True, return:
            beta_all: (G+p+1)*G matrix of coefficients
            robust_se: (G+p+1)*G matrix of robust standard errors
            t_stats: (G+p+1)*G matrix of t statistics
            p_values: (G+p+1)*G matrix of p values
        else return:
            beta_all: (G+p+1)*G matrix of coefficients
        """
        if self.X is None:
            X = np.zeros((self.N, self.G + self.p + 1, self.T - 1))
            Y = np.zeros((self.N, self.T - 1))
        else:
            X = np.zeros((self.N, self.G + self.p + 1, self.T))
            Y = np.zeros((self.N, self.T))
        beta_all = np.zeros((self.G + self.p + 1, self.G))
        robust_se = np.zeros((self.G + self.p + 1, self.G))
        t_stats = np.zeros((self.G + self.p + 1, self.G))
        p_values = np.zeros((self.G + self.p + 1, self.G))
        info = np.zeros((3, self.G)) # 用于存储每个组的R2, R2_adj, N_obs
        if self.X is None:  
            # 如果使用自回归
            '''
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
            '''
            X = np.zeros((self.N, self.G + self.p + 1, self.T-1))
            Y = self.Y[:, 1:]
            # 创建group的one-hot矩阵，形状: N*G
            group_onehot = np.eye(self.G)[self.group]  # N*G
            # network * Y
            network_Y = self.network[:, :, :-1] * self.Y[:, :-1]  # N*N*(T-1)
            # 对于每个 i, g, t: sum_j network[i,j,t] * Y[j,t] * group_mask[j,g]
            X[:, :self.G, :] = np.einsum('ijt,jg->igt', network_Y, group_onehot)
            # 自身滞后
            X[:, self.G, :] = self.Y[:, :-1]
            # CV部分
            X[:, self.G+1:, :] = self.CV[:, :, :-1]
        else:
            # 如果不使用自回归
            '''
            for i in range(self.N):
                # 取Network 第i列
                X_i = np.zeros((self.G+self.p+1, self.T))
                Y_i = self.Y[i, :]
                for g in range(self.G):
                    WX = self.network[i, :, :] * self.X[:, :] * np.repeat((self.group == g).reshape(-1, 1), self.T, axis=1)  # N*T
                    X_i[g, :] = np.sum(WX, axis=0)
                X_i[self.G, :] = self.X[i, :]
                X_i[self.G+1:self.G+self.p+1, :] = self.CV[i, :, :]
                X[i, :, :] = X_i
                Y[i, :] = Y_i
            '''
            # 创建group的one-hot矩阵，形状: N*G
            group_onehot = np.eye(self.G)[self.group]  # N*G
            # 初始化X和Y
            X = np.zeros((self.N, self.G + self.p + 1, self.T))
            Y = self.Y.copy()  # 直接拷贝
            # network * X 按组加权求和，完全向量化
            # 对于每个 i, g, t: sum_j network[i,j,t] * X[j,t] * (group[j]==g)
            X[:, :self.G, :] = np.einsum('ijt,jg->igt', self.network * self.X[None, :, :], group_onehot)
            # 自身值
            X[:, self.G, :] = self.X
            # CV部分
            X[:, self.G+1:, :] = self.CV
        for g in range(self.G):
            if np.sum(self.group == g) == 0:
                # 初始化系数的时候不会出现某个组为0的情况
                beta_all[:, g] = self. beta[:, g]
                continue
            Xg = np.concatenate(X[self.group == g],axis=1)  # N_obs*(G+p+1)*(T-1)
            yg = np.concatenate(Y[self.group == g])  # N_obs*(T-1)
            # 计算OLS系数
            beta = np.linalg.solve(Xg @ Xg.T, Xg @ yg)
            beta_all[:, g] = beta
            if final:
                # 如果是最终结果，计算稳健标准误、t统计量和p值
                robust_se_g, t_stats_g, p_values_g, R2_g, R2_adj_g, n_g = self.cul_OLS_result(Xg, yg, beta)
                robust_se[:, g] = robust_se_g
                t_stats[:, g] = t_stats_g
                p_values[:, g] = p_values_g
                info[0, g] = R2_g
                info[1, g] = R2_adj_g
                info[2, g] = n_g
        if final:
            # 返回最终结果
            return beta_all, robust_se, t_stats, p_values, info
        else:
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
        '''
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
        '''
        # N*N 系数矩阵
        self.coef = beta[np.ix_(group, group)].T   # (N,N)
        # N 长度的动量系数
        self.v_coef = beta[self.G, group]         # (N,)
        # N*p 的CV系数
        self.gamma_coef = beta[self.G+1:self.G+self.p+1, group].T  # (N,p)
        # N*N*T 的网络系数矩阵
        self.network_coef = self.network * self.coef[:, :, None]   # (N,N,T)
        #计算每一天的 Yt = v@Yt-1 + network_coef@Yt-1 + gamma@CV + eps，注意是矩阵乘法
        if self.X is None:
            # 如果使用自回归
            '''
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
            '''
            # momentum 部分
            momentum = self.v_coef[:, None] * self.Y[:, :-1]  # (N, T-1)
            # network 部分
            network_part = np.einsum('ijn,jn->in', self.network_coef[:, :, :-1], self.Y[:, :-1])  # (N, T-1)
            # CV 部分
            CV_part = np.sum(self.gamma_coef[:, :, None] * self.CV[:, :, :-1], axis=1)  # (N, T-1)
            # 合并
            Y_est = momentum + network_part + CV_part
            # loss
            loss = np.mean((Y_est - self.Y[:, 1:])**2)
        else:
            # 如果不使用自回归
            '''
            Y_est = np.zeros((self.N, self.T))
            for t in range(self.T):
                # 计算动量部分
                momentum = self.v_coef * self.X[:, t]
                # 计算网络部分
                network_part = self.network_coef[:, :, t] @ self.X[:, t]
                # 计算CV部分（先乘再求和）
                CV_part = np.sum(self.gamma_coef * self.CV[:, :, t], axis=1)
                # 计算当天的Yt
                Y_est[:, t] = momentum + network_part + CV_part
            loss = np.sum((Y_est - self.Y) ** 2) / (self.N * self.T)
            '''
            momentum = self.v_coef[:, None] * self.X  # (N, T)
            network_part = np.einsum('ijn,jn->in', self.network_coef, self.X)  # (N, T)
            CV_part = np.einsum('ip,ipt->it', self.gamma_coef, self.CV)  # (N, T)
            Y_est = momentum + network_part + CV_part
            loss = np.mean((Y_est - self.Y) ** 2)
        return loss    
    
    def update_para_group(self,leave=True):
        """ Update the group labels based on the loss function
        return:
        update_count: int, the number of nodes whose group labels are updated
        """
        update_count = 0
        if self.group is None:
            raise ValueError("Group is not initialized, Run k_means_clustering() first")
        # update beta
        self.beta = self.cul_para_OLS()
        Node_loss = []
        for i in tqdm(range(self.N), leave=leave):
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
                Node_loss.append(np.min(i_group_loss))
        # 查看有几个组的节点数目是0 or 1
        Node_loss = np.array(Node_loss)
        for g in range(self.G):
            if np.sum(self.group == g) <= 1:
                # 选择loss最大的int(2n/G)个节点中随机一半加入该组
                candidates = np.argsort(Node_loss)[-2*self.N//(self.G):]
                np.random.shuffle(candidates)
                self.group[candidates[:len(candidates)//2]] = g
                update_count += len(candidates)//2
                Node_loss[candidates] = 0 # 避免重复选择
                print(f"Group {g} is empty, add {len(candidates)//2} nodes to it.")
        return update_count

    def update_all_para(self, max_iter=100, leave=True):
        """ Update all parameters until convergence or max_iter reached
        return:
        group: N vector of group labels
        beta: (G+p+1)*G matrix of coefficients
        """
        for epoch in range(max_iter):
            update_count = self.update_para_group(leave=leave)
            if leave:
                print(f"Epoch {epoch}: {update_count} nodes updated")
            if update_count == 0:
                break
        return self.group, self.beta
    
    class GNAR_result:
        """ A class to store the results of the GNAR model
        Attributes:
            beta: (G+p+1)*G matrix of coefficients
            robust_se: (G+p+1)*G matrix of robust standard errors
            t_stats: (G+p+1)*G matrix of t statistics
            p_values: (G+p+1)*G matrix of p values
            group: N vector of group labels
            info: 3*G matrix of R2, R2_adj, N_obs
        """
        def __init__(self, beta, robust_se, t_stats, p_values, group, group0, info, G, N, T, p):
            self.beta = beta   # (G+p+1)*G matrix of coefficients
            self.robust_se = robust_se # (G+p+1)*G matrix of robust standard errors
            self.t_stats = t_stats # (G+p+1)*G matrix of t statistics
            self.p_values = p_values # (G+p+1)*G matrix of p values
            self.group = group # N vector of group labels
            self.group0 = group0 # Initial group labels before clustering
            self.info = info # 3*G matrix of R2, R2_adj, N_obs
            self.G = G # Number of groups
            self.N = N # Number of nodes
            self.T = T # Number of time periods
            self.p = p # Number of covariates
        
        def summary(self):
            """ Print a summary of the GNAR model results """
            print("\033[1;31mGNAR Model Results:\033[0m")
            print(f"N of Groups: {self.G:<5}, Nodes: {self.N:<5}, Time Periods: {self.T:<5}, Covariates: {self.p:<5}")
            for g in range(self.G):
                print("=========================== Group {} =============================".format(g))
                print(f"\033[1;34mNodes_n\033[0m: {np.where(self.group == g)[0].shape[0]:<5}, \033[1;34mObs_n\033[0m: {self.info[2, g]:<5.0f}, \033[1;34mR2\033[0m: {self.info[0, g]:<10.4f}, \033[1;34mR2_adj\033[0m: {self.info[1, g]:<10.4f}")
                print(f"\033[1;32m{f'Var':<15}{f'Coef':<15}{f'Robust SE':<15}{f't-stat':<15}{f'p-value':<15}\033[0m")
                for i in range(self.G):
                    print(f"{f'beta_g{g}g{i}':<15}{self.beta[i, g]:<15.4f}{self.robust_se[i, g]:<15.4f}{self.t_stats[i, g]:<15.4f}{self.p_values[i, g]:<15.4f}")
                print(f"{f'v_g{g}':<15}{self.beta[self.G, g]:<15.4f}{self.robust_se[self.G, g]:<15.4f}{self.t_stats[self.G, g]:<15.4f}{self.p_values[self.G, g]:<15.4f}")
                for i in range(self.p):
                    print(f"{f'gamma_g{g}_{i}':<15}{self.beta[self.G + i, g]:<15.4f}{self.robust_se[self.G + i, g]:<15.4f}{self.t_stats[self.G + i, g]:<15.4f}{self.p_values[self.G + i, g]:<15.4f}")

    def fit(self, method='networkeffect', max_iter=100, time_varying=False, leave=True):
        """ Fit the GNAR model
        method: str, the method for k-means clustering, default is 'networkeffect', you can also use 'momentum' or 'fixedeffect', and 'complete' is a test method!!!
        time_varying: bool, whether the model is time-varying, default is False
        max_iter: int, the maximum number of iterations for updating parameters, default is 100
        """
        self.group0 = self.k_means_clustering(method=method, time_varying=time_varying)
        self.update_all_para(max_iter=max_iter, leave=leave)
        self.beta, robust_se, t_stats, p_values, info = self.cul_para_OLS(final=True)
        res = self.GNAR_result(beta=self.beta, robust_se=robust_se,
                                t_stats=t_stats, p_values=p_values,
                                group=self.group, group0=self.group0,
                                info=info, G=self.G,
                                N=self.N, T=self.T, p=self.p)
        return res

    class GNAR_GIC_result:
        """ A class to store the GIC results of the GNAR model
        Attributes:
            G_list: list of G values tested
            GIC: dictionary of GIC values for each G
            res: dictionary of GNAR_result objects for each G
            loss: dictionary of loss values for each G
            best_G: the best G value with the minimum GIC
            best_GIC: the minimum GIC value
        """
        def __init__(self):
            self.G_list = []
            self.GIC = {}
            self.res = {}
            self.loss = {}
            self.best_G = None
            self.best_GIC = None
        
        def summary(self):
            """ Print a summary of the GIC results """
            print(f"\033[1;31m GIC: best G = {self.best_G}\033[0m")
            self.res[self.best_G].summary()

        def plot(self,figsize=(10, 6), dpi=100):
            """ Plot the GIC values for each G """
            import matplotlib.pyplot as plt
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(self.G_list, list(self.GIC.values()), marker='o')
            plt.xlabel('Number of Groups (G)')
            plt.ylabel('GIC Value')
            plt.title('GIC Values for Different Number of Groups')
            plt.grid()
            plt.show()

    def fit_GIC(self,method='networkeffect',max_G=8,max_iter=100,time_varying=False):
        GIC_res = self.GNAR_GIC_result()
        GIC_res.G_list = list(range(1, max_G + 1))
        for G in GIC_res.G_list:
            self.G = G
            res = self.fit(method=method, max_iter=max_iter, 
                           time_varying=time_varying, leave=False)
            # 计算每个G的loss
            loss = self.loss_function()
            out_degree = np.sum(self.network, axis=0)  # 每个节点的出度
            n_90 = np.percentile(out_degree, 90)  # 90%分位数
            lambda_NT = ((self.N**0.1) * (self.T**(-0.5))) / (2*min(n_90,10))
            GIC = np.log(loss) + lambda_NT*G
            GIC_res.GIC[G] = GIC
            GIC_res.res[G] = res
            GIC_res.loss[G] = loss
            print(f"G={G}, GIC={GIC:.4f}, loss={loss:.4f}, lambda_NT={lambda_NT:.4f}")
        # 找到最小的GIC
        GIC_res.best_G = min(GIC_res.GIC, key=GIC_res.GIC.get)
        GIC_res.best_GIC = GIC_res.GIC[GIC_res.best_G]
        self.G = GIC_res.best_G
        print(f"Best G={GIC_res.best_G}, GIC={GIC_res.best_GIC:.4f}, back to G={self.G}")
        return GIC_res

