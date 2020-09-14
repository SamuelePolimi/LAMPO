import numpy as np
import scipy
from torch.distributions import multivariate_normal, lowrank_multivariate_normal
from scipy.stats import multivariate_normal as scipy_normal

from mppca.mixture_ppca import MPPCA
from functools import reduce  # Required in Python 3
import operator

import torch


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def sum_logs(X, axis=0, mul=1.):
    """
    X is in log-space, we will return the sum in log space
    :param X:
    :param axis:
    :return:
    """
    x_max = torch.max(X, dim=axis).values
    X_exp = mul * torch.exp(X-x_max)
    return x_max + torch.log(torch.sum(X_exp, dim=axis))


def sum_logs_np(X, axis=0, mul=1.):
    """
    X is in log-space, we will return the sum in log space
    :param X:
    :param axis:
    :return:
    """
    x_max = np.max(X, axis=axis)
    X_exp = mul * np.exp(X-x_max)
    return x_max + np.log(np.sum(X_exp, axis=axis))


class RLModel:
    """
    Probabilistic and differentiable model for improving the latent space
    """

    def __init__(self, mppca: MPPCA, context_dim, normalize=True, kl_type="forward", kl_bound=0.5):

        torch.set_default_tensor_type('torch.DoubleTensor')

        # parameters
        self.log_base_pi = torch.tensor(mppca.log_pi, requires_grad=True)
        self.mu = torch.zeros((mppca.n_components, mppca.latent_dimension), requires_grad=True)  # torch.from_numpy(mu)
        self._base_diag_Sigma = torch.tensor(np.ones((mppca.log_pi.shape[0], self.mu.shape[1])), requires_grad=True)

        self._last_log_pi = self.get_log_pi(self.log_base_pi).detach().clone()
        self._last_Sigma = self.get_Sigma(self._base_diag_Sigma).detach().clone()
        self._last_mu = self.mu.detach().clone()

        # list of parameters
        self._parameter_list = [self.log_base_pi, self.mu, self._base_diag_Sigma]

        self.W = torch.from_numpy(mppca.linear_transform)
        # self.log_pi = self.log_base_pi - sum_logs(self.log_base_pi)

        self.means = torch.from_numpy(mppca.means)
        self.sigma_sq = torch.from_numpy(mppca.sigma_squared)
        self._state_dim = mppca.linear_transform.shape[1]
        self._context_dim = context_dim
        self._action_dim = self._state_dim - context_dim
        self._n_cluster = mppca.log_pi.shape[0]
        self._latent_dim = self.mu.shape[1]

        # Dataset
        self._w = torch.zeros((0, self._action_dim))
        self._z = torch.zeros((0, self._latent_dim))
        self._k = torch.zeros(0, dtype=torch.long)
        self._c_data = torch.zeros((0, self._context_dim))
        self._r = torch.zeros(0)
        self._rho_den = []

        self._normalize = normalize
        self.__kl_type = kl_type

        self._inner_x = None
        self._f = None
        self._f_grad = None
        self._g = None
        self._g_grad = None

        self.performance = None

        self.avg_entropy = 0.
        self._kl_bound = kl_bound
        # self._optimizer = torch.optim.Adam(params=[self.log_base_pi, self.mu, self._base_diag_Sigma], lr=1E-2)
        self.preprocess()

    def save_new_policy(self):
        self._last_log_pi = self.get_log_pi(self.log_base_pi).detach().clone()
        self._last_Sigma = self.get_Sigma(self._base_diag_Sigma).detach().clone()
        self._last_mu = self.mu.detach().clone()

    def add_dataset(self, w, z, k, c, r):
        k = k.astype(np.long)
        w, z, k, c, r = map(torch.from_numpy, [w, z, k, c, r])

        self._w = torch.cat([self._w, w], 0)
        self._z = torch.cat([self._z, z], 0)
        self._k = torch.cat([self._k, k], 0)
        self._c_data = torch.cat([self._c_data, c], 0)
        self._r = torch.cat([self._r, r], 0)

        with torch.no_grad():
            # TODO should already be detached
            self._rho_den.append(self._log_policy(z, k, c,
                                    self._last_log_pi, self._last_mu, self._last_Sigma).detach())

    def preprocess(self):
        self._Omega = self.W[:, self._context_dim:, :]
        self._C = self.W[:, :self._context_dim, :]
        self._omega = self.means[:, self._context_dim:]
        self._c = self.means[:, :self._context_dim]

    def update_variables(self):
        self.log_pi = self.get_log_pi(self.log_base_pi)
        self.Sigma = self.get_Sigma(self._base_diag_Sigma)

    def get_Sigma(self, sigma_base):
        """
        Retreive the Sigma tensor from its base value
        :param sigma_base:
        :return:
        """
        return torch.stack([torch.diag(s ** 2) for s in sigma_base])

    def get_log_pi(self, log_pi_base):
        """
        Retreive the log_pi tensor from its base value
        :param log_pi_base:
        :return:
        """
        return log_pi_base - sum_logs(log_pi_base)

    def _get_gradient(self):
        grads = []
        for param in [self.log_base_pi, self.mu, self._base_diag_Sigma]:
            grads.append(param.grad.detach().numpy().ravel())
        return np.concatenate(grads)

    def set_parameters(self, base_log_pi, mu, base_sigma):
        """
        set the inner (base) parameters with new values
        :param base_log_pi:
        :param mu:
        :param base_sigma:
        :return:
        """
        self.log_base_pi.data = base_log_pi
        self.mu.data = mu
        self._base_diag_Sigma.data = base_sigma

    def _set_x(self, x):
        """
        unpack the values of x in the parameter vector
        :param x:
        :return:
        """
        i = 0
        for param_i in self._parameter_list:
            shape = param_i.shape
            param_len = prod(shape)
            # slice out a section of this length
            param = x[i:i + param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*shape)
            param_i.data = torch.from_numpy(param)
            # update index
            i += param_len

    def _get_x(self):
        """
        Return a unidimensional vector of parameters
        :return:
        """
        ret = []
        for param_i in self._parameter_list:
            ret.append(np.ravel(param_i.data.detach().numpy()))
        return np.concatenate(ret)

    def _log_responsabilities(self, c, log_pi, mu, Sigma):
        """
        Compute the log probability for each cluster given a context c.
        :param c: context
        :return:
        """
        n = c.shape[0]

        stack = []
        for i in range(self._n_cluster):
                stack.append(self._log_p_c_k(c, torch.tensor([i]).repeat(n), mu.squeeze(-1), Sigma) + log_pi[i])
        log_p_c = torch.stack(stack)
        norm = sum_logs(log_p_c, axis=0)
        return log_p_c - norm.view(1, -1)

    def generate_full(self, c, noise=True, isomorphic_noise=False):
        """
        Sample w, z, k from a vector of contextes.
        :param c:
        :param noise:
        :param isomorphic_noise:
        :return:
        """
        c = torch.from_numpy(c)
        p = np.exp(self._log_responsabilities(c, self._last_log_pi, self._last_mu, self._last_Sigma).detach().numpy()).T
        k = [np.random.choice(range(self._n_cluster),
                             p=p_i) for p_i in p]
        O, o = self._Omega[k], self._omega[k]

        mean, cov = self._get_mean_cov_z(c, k, self._last_mu, self._last_Sigma)  # TODO: check single param here
        if noise:
            m = multivariate_normal.MultivariateNormal(mean, cov)
            z = m.sample()
        else:
            z = mean.unsqueeze(0)

        if isomorphic_noise:
            m = multivariate_normal.MultivariateNormal((O @ z.unsqueeze(2)).squeeze(2) + o, O @ cov @ O.transpose(1, 2) +
                    self.sigma_sq[k].unsqueeze(1).unsqueeze(2) * torch.eye(self._action_dim).expand(len(k), self._action_dim, self._action_dim))
            w = m.sample()
        else:
            w = O @ z.unsqueeze(2) + o.unsqueeze(2)

        return w.detach().numpy().squeeze(), z.detach().numpy().squeeze(), np.array(k).squeeze()

    def _get_mean_cov_z(self, c, k, mu, Sigma):
        """
        Compute the mean and the covariance of z given c, k (and policy parameters)
        :param c:
        :param k:
        :param mu:
        :param Sigma:
        :return:
        """
        mu_, S, s = mu[k].unsqueeze(2), Sigma[k], self.sigma_sq[k]

        C, _c = self._C[k], self._c[k]

        B = torch.inverse(s.view(-1, 1, 1) * torch.inverse(S) + torch.transpose(C, -2, -1) @ C)
        D = B @ torch.transpose(C, -2, -1)
        d = s.view(-1, 1, 1) * B @ torch.inverse(S) @ mu_ - B @ torch.transpose(C, -2, -1) @ _c.unsqueeze(2)

        mean = D @ c.unsqueeze(2) + d
        cov = s.view(-1, 1, 1) * B

        return mean.squeeze(), cov

    def _get_cluster_mean_cov_z(self, c, mu, sigma):
        """
        Given a context c, it returns the parameters of the gaussian distrubiton p(z|c, k_i) for all i.
        :param c: vector of context (n_samples, context_dim)
        :param mu: vector of mu parameters
        :param sigma: vector of Sigma parameters
        :return:
        """
        means = []
        covs = []
        for k in range(self._n_cluster):
            mean, cov = self._get_mean_cov_z(c, torch.tensor([k]).repeat(c.shape[0]), mu, sigma)
            means.append(mean)
            covs.append(cov)
        return torch.stack(means).transpose(0, 1), torch.stack(covs).transpose(0, 1)

    def _log_p_c_k(self, c, k, mu, Sigma):
        """
        compute the log-probability of the context c given the cluster k.
        :param c: xontext
        :param k: cluster
        :return:
        """

        n = k.shape[0]
        mean = self._C[k] @ mu[k].unsqueeze(2) + self._c[k].unsqueeze(2)
        base_cov = self.sigma_sq[k].view(-1, 1, 1) \
                   * torch.eye(self._context_dim, dtype=torch.float64).unsqueeze(0).repeat(n, 1, 1)
        cov = base_cov + self._C[k] @ Sigma[k] @ torch.transpose(self._C[k], -2, -1)
        # TODO before wat squeeze()
        m = multivariate_normal.MultivariateNormal(mean.squeeze(-1), cov)
        return m.log_prob(c)

    def get_params_z_given_k(self, k, mu, Sigma):
        """
        p(z | k) = N(.|mu[k), Sigma[k])
        :param k:
        :param mu:
        :param Sigma:
        :return:
        """
        return mu[k], Sigma[k]

    def get_params_c_given_z_k(self, z, k):
        """
        p(c | z, k) = N(.|\Omega @ z, \sigma^2_k I)
        :param z:
        :param k:
        :param mu:
        :param sigma:
        :return:
        """
        return (self._C[k] @ z.unsqueeze(2)).squeeze(-1) + self._c[k], \
               self.sigma_sq[k].unsqueeze(1).unsqueeze(2) * torch.eye(self._context_dim).expand([z.shape[0],
                                                                            self._context_dim, self._context_dim])

    def get_params_marginal(self, A, b, L_inv, mu, Lambda_inv):
        """
        p(x) = N(x | mu, Lambda_inv)
        p(y | x) = N(y | Ax + b, L_inv)
        return E[y], Cov[y, y]
        :param A:
        :param b:
        :param L_inv:
        :param mu:
        :param Lambda_inv:
        :return:
        """
        return (A @ mu.unsqueeze(2)).squeeze(-1) + b, L_inv + A @ Lambda_inv @ torch.transpose(A, 1, 2)

    def get_params_c_given_k(self, k, mu, sigma):
        A = self._C[k]
        b = self._c[k]
        L_inv = self.sigma_sq[k].unsqueeze(1).unsqueeze(2) * torch.eye(self._context_dim).expand(
            [k.shape[0], self._context_dim, self._context_dim])

        m = mu[k]
        Lambda_inv = sigma[k]
        return self.get_params_marginal(A, b, L_inv, m, Lambda_inv)

    def _log_policy(self, z, k, c, log_pi, mu, Sigma):
        z, k, c, log_pi, mu, Sigma = [torch.from_numpy(x) if type(x) is np.ndarray else x for x in
                                      [z, k, c, log_pi, mu, Sigma]]

        # mu = mu.squeeze(3)
        log_pi_k = log_pi[k]
        log_z_given_k = multivariate_normal.MultivariateNormal(mu[k], Sigma[k]).log_prob(z)
        m, s = self.get_params_c_given_z_k(z, k)
        log_c_given_z_k = multivariate_normal.MultivariateNormal(m, s).log_prob(c)
        log_c_z_k = log_c_given_z_k + log_z_given_k + log_pi_k
        stack = []
        for i in range(self._n_cluster):
            k_i = torch.tensor([i]).expand([k.shape[0]])
            m, s = self.get_params_c_given_k(k_i, mu, Sigma)
            log_c_given_k = multivariate_normal.MultivariateNormal(m, s).log_prob(c)
            log_pi_k = log_pi[k_i]
            stack.append(log_pi_k + log_c_given_k)
        log_c = sum_logs(torch.stack(stack, 0), axis=0)
        return log_c_z_k - log_c

    def _get_gradient_from_torch(self, f: torch.Tensor):
        """
        Get the gradient of f w.r.t. the policy's parameters.
        :param f: The parametric function.
        :return: the gradient.
        """
        f.backward()
        g = self._get_gradient()
        self.zero_grad()
        return g

    def get_objective(self):
        log_p = self._log_policy(self._z, self._k, self._c_data,
                                 self.get_log_pi(self.log_base_pi), self.mu, self.get_Sigma(self._base_diag_Sigma))

        rho_num = log_p
        rho_den = torch.cat(self._rho_den)

        if self._normalize:
            w = torch.exp(rho_num - rho_den - sum_logs(rho_num - rho_den))
        else:
            w = torch.exp(rho_num - rho_den) / self._k.shape[0]

        ret = torch.sum(self._r*w)
        return self._get_gradient_from_torch(ret), ret.detach().numpy()

    def zero_grad(self):
        self._base_diag_Sigma.grad.zero_()
        self.mu.grad.zero_()
        self.log_base_pi.grad.zero_()

    def get_kl_regularization(self):
        """
        Add the forward KL entropy bound
        :param c: vector of context (n_samples x context_dim)
        :return:
        """
        c = self._c_data
        log_pi_reg = self.get_log_pi(self.log_base_pi)
        mu_reg = self.mu
        sigma_reg = self.get_Sigma(self._base_diag_Sigma)
        log_pi_a = self._log_responsabilities(c, self._last_log_pi, self._last_mu, self._last_Sigma)
        log_pi_b = self._log_responsabilities(c, log_pi_reg, mu_reg, sigma_reg)

        mu_a, sigma_a = self._get_cluster_mean_cov_z(c, self._last_mu, self._last_Sigma)
        mu_b, sigma_b = self._get_cluster_mean_cov_z(c, mu_reg, sigma_reg)

        if self.__kl_type == "forward":
            kl = self._get_kl(mu_b, sigma_b, log_pi_b.T, mu_a, sigma_a, log_pi_a.T)
        elif self.__kl_type == "reverse":
            kl = self._get_kl(mu_a, sigma_a, log_pi_a.T, mu_b, sigma_b, log_pi_b.T)
        else:
            raise Exception("Value '%s' not known. Kl must be 'none', 'forward', 'reverse'." % self.__kl_type)

        kl_mean = torch.mean(kl)
        return self._get_gradient_from_torch(kl_mean), kl_mean.detach().numpy()

    def _get_kl(self, mu_a, sigma_a, log_pi_a, mu_b, sigma_b, log_pi_b):
        """
        Given a set of mixture distributions, compute the kl between them.
        :param mu_a:
        :param sigma_a:
        :param pi_a:
        :param mu_b:
        :param sigma_b:
        :param pi_b:
        :return:
        """
        H = - self._get_parallel_cross_entropy(mu_a, sigma_a, log_pi_a, mu_a, sigma_a, log_pi_a)
        self.avg_entropy = torch.mean(H).detach().numpy()
        Cr_H = self._get_parallel_cross_entropy(mu_a, sigma_a, log_pi_a, mu_b, sigma_b, log_pi_b)
        return H + Cr_H

    def _get_parallel_cross_entropy(self, mu_a, sigma_a, log_pi_a, mu_b, sigma_b, log_pi_b):
        """
        Compute -\int p(a) log p(b)
        where p(a) and p(b) are Gaussian mixtures
        :param mu_a:
        :param sigma_a:
        :param log_pi_a:
        :param mu_b:
        :param sigma_b:
        :param log_pi_b:
        :return:
        """
        cross_entropy_multinomials = -torch.einsum("ij,ij->i", torch.exp(log_pi_a), log_pi_b)
        d = mu_a.shape[2]
        cross_entropy_normals = 0.
        for i in range(mu_a.shape[1]):
            t_1 = d * np.log(2*np.pi)
            t_2 = torch.log(torch.det(sigma_b[:, i]))
            L_inv = torch.inverse(sigma_b[:, i])
            t_3 = torch.einsum("ijj->i", L_inv @ (torch.einsum("ij,ik->ijk",mu_a[:, i], mu_a[:, i]) + sigma_a[:,i]))
            t_4 = torch.einsum("ij,ij->i", mu_b[:, i], ((L_inv @ mu_a[:, i].unsqueeze(2)).squeeze(-1)))
            t_5 = torch.einsum("ij,ij->i", mu_a[:, i], ((L_inv @ mu_b[:, i].unsqueeze(2)).squeeze(-1)))
            t_6 = torch.einsum("ij,ij->i", mu_b[:, i], ((L_inv @ mu_b[:, i].unsqueeze(2)).squeeze(-1)))
            cross_entropy_normals += torch.exp(log_pi_a[:, i]) * 0.5 * (t_1 + t_2 + t_3 - t_4 - t_5 + t_6)
        return cross_entropy_normals + cross_entropy_multinomials

    def get_f(self, x):
        if self._inner_x is not None:
            if np.equal(x, self._inner_x):
                return self._f
        self._set_x(x)
        self._f_grad, self._f = self.get_objective()
        return -self._f

    def get_f_grad(self, x):
        if self._inner_x is not None:
            if np.equal(x, self._inner_x):
                return self._f_grad
        self._set_x(x)
        self._f_grad, self._f = self.get_objective()
        return -self._f_grad

    def get_g(self, x):
        if self._inner_x is not None:
            if np.equal(x, self._inner_x):
                return self._g
        self._set_x(x)
        self._g_grad, self._g = self.get_kl_regularization()
        return -self._g + self._kl_bound

    def get_g_grad(self, x):
        if self._inner_x is not None:
            if np.equal(x, self._inner_x):
                return self._g_grad
        self._set_x(x)
        self._g_grad, self._g = self.get_kl_regularization()
        return -self._g_grad