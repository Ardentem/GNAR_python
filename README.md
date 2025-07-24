# Python Code for GNAR 

## 1. Grouped Network Vector Autoregression

This Python code implements the Grouped Network Vector Autoregression (GNAR) model.

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

For more details, please refer to [Ardentem's Blog (in Chinese)](https://ardentemwang.com/2025/07/22/GNAR1/#more)

## References:

- Zhu, X., Pan, R., Li, G., Liu, Y., & Wang, H. (2017). [Network Vector Autoregression](http://ibids.cn/pdf/aos2017.pdf). The Annals of Statistics, 45(3), 1096-1123.

- Zhu, X., & Pan, R. (2020). [Grouped network vector autoregression](https://www.jstor.org/stable/26968936). Statistica Sinica, 30(3), 1437-1462.

- Zhu, X., Xu, G., & Fan, J. (2025). [Simultaneous estimation and group identification for network vector autoregressive model with heterogeneous nodes](https://www.sciencedirect.com/science/article/pii/S0304407623002804). Journal of Econometrics, 249, 105564.
