# Python Code for GNAR 

> <font color='red'>**Warning!!**</font>  
> The author is not a professional programmer, and the code is for self-study purposes only. Some methods may not conform to the settings in the original paper, nor have they undergone rigorous mathematical reasoning and proof. **The author does not guarantee the correctness or usability of the code**. Please use with caution.

Code files:
- `estimator.py`: contains the GNAR estimator class for fitting the model and summarizing results
- `simulator.py`: simulation functions to generate Y, Z, W
- `main.ipynb`: generate data and conduct estimation

There is a brief tutorial in `main.ipynb` to demonstrate how to use the functions in this package.

## 1. Grouped Network Vector Autoregression

This Python code implements the Grouped Network Vector Autoregression (GNAR) model ([Zhu et al., 2025](https://www.sciencedirect.com/science/article/pii/S0304407623002804)).

Consider a network with $N$ nodes, indexed by $i=1,\ldots,N$, whose relationships are recorded by the adjacency matrix $A=(a_{ij})\in\{0,1\}^{N\times N}$, where $a_{ij}=1$ indicates that node $i$ follows node $j$, and $a_{ij}=0$ otherwise. By convention, $a_{ii}=0$. For each node $i$, a time series of a continuous variable $\{Y_{it}\}_{t=0}^T$ is observed, along with a set of node-specific covariates $z_i\in\mathbb{R}^p$. In particular, the first component of $z_i$ is always $1$, corresponding to the intercept term.

To capture network heterogeneity, it is assumed that the nodes can be divided into $G$ groups, with homogeneous regression effects within each group. Let $g_i\in\{1,\ldots,G\}$ denote the group membership of node $i$. The GNAR model is formulated as:

$$\begin{equation}
Y_{it} = \sum_{j=1, j\neq i}^N \beta_{g_i g_j} w_{ij} Y_{j,(t-1)} + \nu_{g_i} Y_{i,(t-1)} + z_i^\top \zeta_{g_i} + \varepsilon_{it}
\end{equation}$$

where $w_{ij}=n_i^{-1} a_{ij}$, $n_i=\sum_{j=1}^N a_{ij}$ denotes the out-degree of node $i$, and $\varepsilon_{it}$ is an independent and identically distributed random noise with mean $0$ and variance $\sigma^2$. All model parameters and node group memberships $g_i$ are to be estimated.


## 2. Time-Varying GNAR
This section additionally considers time variation.
Specifically, we further assume that both the network structure and the covariates are time-varying, and the model becomes:

$$\begin{equation}
Y_{it} = \sum_{j=1, j\neq i}^N \beta_{g_i g_j} w_{ij}(t) Y_{j,(t-1)} + \nu_{g_i} Y_{i,(t-1)} + z_{it}^\top \zeta_{g_i} + \varepsilon_{it}
\end{equation}$$

Under this new setting, the original K-means grouping method is no longer applicable, since both the network structure and covariates are time-varying. Therefore, the fixed effects cannot be eliminated simply by demeaning, and an improved K-means initialization is required:

For more details of the Time-Varying GNAR and the improved K-means initialization, please refer to [Ardentem's Blog (1) (in Chinese)](https://ardentemwang.com/2025/07/22/GNAR1/#more)

## 3. Normal Regression with Grouped Network Effect

In addition to the Autoregression Model, this code also implements a normal regression model with network effect(e.g., [Zacchia, 2020](https://academic.oup.com/restud/article-abstract/87/4/1989/5505452?login=false)). The model is formulated as:

$$\begin{equation}
Y_{it} = \sum_{j=1, j\neq i}^N \beta_{g_i g_j} w_{ij}(t) X_{j,t} + \nu_{g_i} Y_{i,t} + z_{it}^\top \zeta_{g_i} + \varepsilon_{it}
\end{equation}$$

The same estimation method is applied to this model, and the K-means initialization is also improved to accommodate the time-varying nature of the network and covariates.

For more details, please refer to [Ardentem's Blog (2) (in Chinese)](https://ardentemwang.com/2025/08/03/GNAR2/#more)

## 4. Use GIC to select the number of groups

According to [Zhu et al.(2025)](https://www.sciencedirect.com/science/article/pii/S0304407623002804), the Group Information Criterion (GIC) can be used to select the number of groups in the GNAR model.

$$
\begin{equation}
\mathbb{GIC}_{\lambda_{NT}}(G) = \log \left\{\mathbf{Q}\left(\widehat{\boldsymbol{\theta}}^{(G)}, \widehat{\boldsymbol{\beta}}^{(G)}, \widehat{\mathbf{G}}^{(G)}\right) \right\} + \lambda_{NT} G
\end{equation}
$$

This package sets the tuning parameter $\lambda_{NT}$ as follows, recommended by [Zhu et al.(2025)](https://www.sciencedirect.com/science/article/pii/S0304407623002804)

$$
\begin{equation}
\lambda_{NT} = \dfrac{N^{1/10}T^{-1/2}}{2\min\{10,n_{90}\}}
\end{equation}
$$

## References:

- Zhu, X., Pan, R., Li, G., Liu, Y., & Wang, H. (2017). [Network Vector Autoregression](http://ibids.cn/pdf/aos2017.pdf). The Annals of Statistics, 45(3), 1096-1123.

- Zhu, X., & Pan, R. (2020). [Grouped network vector autoregression](https://www.jstor.org/stable/26968936). Statistica Sinica, 30(3), 1437-1462.

- Zhu, X., Xu, G., & Fan, J. (2025). [Simultaneous estimation and group identification for network vector autoregressive model with heterogeneous nodes](https://www.sciencedirect.com/science/article/pii/S0304407623002804). Journal of Econometrics, 249, 105564.

- Zacchia, P. (2020). [Knowledge spillovers through networks of scientists](https://academic.oup.com/restud/article-abstract/87/4/1989/5505452?login=false). The Review of economic studies, 87(4), 1989-2018.
