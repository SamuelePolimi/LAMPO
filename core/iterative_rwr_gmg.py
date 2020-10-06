import numpy as np
from core.model import sum_logs_np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as scipy_normal
from sklearn.covariance import ledoit_wolf


def stable_cov(cov, reg):
    reg_new = np.copy(reg)
    cov_new = np.copy(cov)
    while True:
        try:
            np.linalg.inv(cov_new)
            break
        except:
            print("singular!")
            cov_new += reg_new
            reg_new *= 2.
    return cov_new


class IRWRGMM:
    """
    Iterative Reward Weighted Responsability Gaussian Mixture Model.
    Colome and Torras 2018.
    """

    def __init__(self, n_componente=1, tol=1E-5, n_init=100, max_iter=100, discount=0.98, cov_regularization=1E-15):
        self._n_components = n_componente
        self._tol = tol
        self._data = None
        self._dim = None

        self._n_i = 0
        self._mus = None
        self._covs = None
        self._log_pi = None

        self._n_init = n_init
        self._max_iter = max_iter

        self._discount = discount

        self._reg = cov_regularization

    def _initialize(self, X):
        n_samples, observed_dimensions = X.shape
        kmeans = KMeans(self._n_components, n_init=self._n_init)
        lab = kmeans.fit(X).predict(X)
        self._covs = []
        for i in range(self._n_components):
            cl_indxs = np.where(lab == i)[0]
            rnd_indxs = np.random.choice(range(n_samples), size=5)
            indx = np.concatenate([cl_indxs, rnd_indxs])
            # Avoid non-singular covariance
            self._covs.append(ledoit_wolf(X[indx])[0])
        self._pi = np.ones(self._n_components) / self._n_components
        self._log_pi = np.log(self._pi)
        self._mus = np.array(kmeans.cluster_centers_)

    def fit_new_data(self, X, w):
        """

        :param X: (n_samples x dim)
        :param w: (n_samples)
        :return:
        """
        first = False
        if self._mus is None:
            first = True
            self._initialize(X)
        # w = w/np.sum(w)
        old_log_likelihood = np.inf
        log_resp, log_likelihood = self.get_log_responsability(X, w)
        it = 0
        old_mu = np.copy(self._mus)
        old_cov = np.copy(self._covs)
        old_n_i = np.copy(self._n_i)
        reg = self._reg * np.eye(X.shape[1])
        while np.abs(old_log_likelihood - log_likelihood) > self._tol and it < self._max_iter:
            print("iter", it, log_likelihood)
            n_i = []
            for i in range(self._n_components):
                d = w * np.exp(log_resp[i])
                if first:
                    n = np.sum(d)
                    n_i.append(n)
                    self._mus[i] = np.einsum("i,ij->j", d, X)/n                             # eq 20
                    Y = X - self._mus[i]
                    cov = np.einsum('k,ki,kj->ij', d, Y, Y)
                    self._covs[i] = stable_cov(cov/n, reg)                                  # eq 21
                else:
                    n = np.sum(d) + old_n_i[i]                                                   # eq 25
                    n_i.append(n)
                    if np.sum(d) >= 1E-10:
                        self._mus[i] = (old_n_i[i]*old_mu[i] + np.einsum("i,ij->j", d, X))/n       # eq 27
                        Y = X - self._mus[i]#np.einsum("i,ij->j", d, X)/np.sum(d)  # np.einsum("i,ij->j", d, X)/np.sum(d) #self._mus[i]
                        cov = np.einsum('k,ki,kj->ij', d, Y, Y)
                        self._covs[i] = stable_cov(old_n_i[i]/n * old_cov[i] + cov/n,
                                                   reg)
                                        # eq 21

            self._n_i = np.copy(n_i)
            # print("n_i", self._n_i)
            # print("n", np.sum(self._n_i))
            self._log_pi = np.log(np.array(n_i)) - np.log(np.sum(n_i))                            # eq 22
            old_log_likelihood = np.copy(log_likelihood)
            log_resp, log_likelihood = self.get_log_responsability(X, w)
            it += 1
        self._n_i = self._n_i * self._discount                                              # eq 29

    def get_log_responsability(self, X, w):
        log_p = []
        for i in range(self._n_components):
            dist = scipy_normal(self._mus[i], self._covs[i], allow_singular=True)
            log_p.append(dist.logpdf(X) + self._log_pi[i])
        # log_p = np.log(np.exp(log_p) + 1E-10)  # avoid collapse
        z = sum_logs_np(log_p, axis=0)
        return np.array(log_p) - z, np.sum(w*z)/np.sum(w)

    def predict(self, x, dim):
        mus = []
        covs = []
        resp = []
        for i in range(self._n_components):
            cov_xx = self._covs[i][:dim, :dim]
            cov_yy = self._covs[i][dim:, dim:]
            cov_xy = self._covs[i][:dim, dim:]
            mu_x = self._mus[i][:dim]
            mu_y = self._mus[i][dim:]
            cov_xx_i = np.linalg.inv(cov_xx)
            new_mu = mu_y + cov_xy.T @ cov_xx_i @ (x - mu_x)
            new_cov = cov_yy - cov_xy.T @ cov_xx_i @ cov_xy
            mus.append(new_mu)
            covs.append(new_cov)
            gauss = scipy_normal(mu_x, cov_xx, allow_singular=True)
            resp = gauss.logpdf(x) + self._log_pi

        select_p = np.exp(np.array(resp) - sum_logs_np(resp))
        cluster = np.random.choice(range(self._n_components), p=select_p/np.sum(select_p))
        return np.random.multivariate_normal(mus[cluster], covs[cluster]), cluster


