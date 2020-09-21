import numpy as np
import scipy
from torch.distributions import multivariate_normal, lowrank_multivariate_normal
from scipy.stats import multivariate_normal as scipy_normal

from mppca.mixture_ppca import MPPCA
from herl.dict_serializable import DictSerializable
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


class RLModel(DictSerializable):
    """
    Probabilistic and differentiable model for improving the latent space
    """

    load_fn = DictSerializable.get_numpy_load()

    torch_vars = ["log_base_pi", "mu", "_base_diag_Sigma"]
    lists = ["_rho_den", "_parameter_list"]

    def __init__(self, mppca: MPPCA, context_dim, normalize=True, kl_type="forward", kl_bound=0.5, kl_bound_context=1., kl_reg=0.01):

        torch.set_default_tensor_type('torch.DoubleTensor')

        if mppca is not None:

            self.log_base_pi = torch.tensor(mppca.log_pi, requires_grad=True)
            self.mu = torch.zeros((mppca.n_components, mppca.latent_dimension), requires_grad=True)  # torch.from_numpy(mu)
            self._base_diag_Sigma = torch.tensor(np.ones((mppca.log_pi.shape[0], self.mu.shape[1])), requires_grad=True)

            self._last_log_pi = self.get_log_pi(self.log_base_pi).detach().clone()
            self._last_Sigma = self.get_Sigma(self._base_diag_Sigma).detach().clone()
            self._last_mu = self.mu.detach().clone()

            self._first_log_pi = self.get_log_pi(self.log_base_pi).detach().clone()
            self._first_Sigma = self.get_Sigma(self._base_diag_Sigma).detach().clone()
            self._first_mu = self.mu.detach().clone()

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
            self._x_g = None
            self._x_f = None
            self._x_h = None

            self._f = None
            self._f_grad = None

            self._g = None
            self._g_grad = None

            self._h = np.array(0.)
            self._h_grad = None

            self.performance = None

            self.avg_entropy = 0.

            self._kl_bound = kl_bound
            self._kl_bound_context = kl_bound_context
            self._kl_reg = kl_reg

            self.current_context_kl = 0.
            # self._optimizer = torch.optim.Adam(params=[self.log_base_pi, self.mu, self._base_diag_Sigma], lr=1E-2)
            self.preprocess()

        DictSerializable.__init__(self, DictSerializable.get_numpy_save())

    @staticmethod
    def load_from_dict(**kwargs):
        model = RLModel(None, kwargs["_context_dim"])
        model.__dict__ = kwargs
        for k in kwargs.keys():
            # TODO: don't know if this is all necessary
            if k in RLModel.torch_vars:
                model.__dict__[k] = torch.tensor(kwargs[k], requires_grad=True)
            elif k in RLModel.lists:
                model.__dict__[k] = [torch.tensor(e) for e in kwargs[k].tolist()]
            else:
                try:
                    model.__dict__[k] = torch.tensor(kwargs[k])
                except:
                    pass
        model._parameter_list = [model.log_base_pi, model.mu, model._base_diag_Sigma]
        model.preprocess()
        return model

    @staticmethod
    def load(file_name: str):
        """

        :param file_name:
        :param domain:
        :return:
        """
        file = RLModel.load_fn(file_name)
        return RLModel.load_from_dict(**file)

    def _get_dict(self):
        ret = {}

        for k in self.__dict__:
            # TODO: don't know if this is all necessary
            if k in RLModel.torch_vars:
                ret[k] = self.__dict__[k].detach().numpy()
            elif k in RLModel.lists:
                ret[k] = [e.detach().numpy() for e in self.__dict__[k]]
                pass
            else:
                try:
                    ret[k] = self.__dict__[k].numpy()
                except:
                    ret[k] = self.__dict__[k]
            #
            # if ret[k] is None:
            #     ret[k] = 0.

        ret = {k: v for k, v in ret.items() if k not in ["save_fn", "_parameter_list", "_RLModel__kl_type"]}
        return ret

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

        self._inner_x = self._get_x()

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

        # mean, prec = self._get_stable_mean_prec_z(c, k, self._last_mu, self._last_Sigma)  # TODO: check single param here
        mean, prec = self._get_mean_prec_z(c, k, self._last_mu, self._last_Sigma)  # TODO: check single param here
        if noise:
            m = multivariate_normal.MultivariateNormal(mean, precision_matrix=prec)
            z = m.sample()
        else:
            z = mean.unsqueeze(0)

        if isomorphic_noise:
            m = multivariate_normal.MultivariateNormal((O @ z.unsqueeze(2)).squeeze(2) + o,
                    self.sigma_sq[k].unsqueeze(1).unsqueeze(2) * torch.eye(self._action_dim).expand(len(k), self._action_dim, self._action_dim))
            w = m.sample()
        else:
            w = O @ z.unsqueeze(2) + o.unsqueeze(2)

        return w.detach().numpy().squeeze(), z.detach().numpy().squeeze(), np.array(k).squeeze()

    def _get_mean_prec_z(self, c, k, mu, Sigma):
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

        E = torch.inverse(S) + torch.inverse(s.view(-1, 1, 1)) * torch.transpose(C, -2, -1) @ C
        B = torch.inverse(s.view(-1, 1, 1) * torch.inverse(S) + torch.transpose(C, -2, -1) @ C)
        D = B @ torch.transpose(C, -2, -1)
        d = s.view(-1, 1, 1) * B @ torch.inverse(S) @ mu_ - D @ _c.unsqueeze(2)

        mean = D @ c.unsqueeze(2) + d
        cov = s.view(-1, 1, 1) * B

        # mean_test, cov_test = self._get_stable_mean_cov_z(c, k, mu, Sigma)
        return mean.squeeze(-1), E

    def _get_stable_mean_prec_z(self, c, k, mu, Sigma):
        """
        Compute the mean and the covariance of z given c, k (and policy parameters)
        using torch.solve instead of matrix inversion (when possible)
        :param c:
        :param k:
        :param mu:
        :param Sigma:
        :return:
        """
        mu_, S, s = mu[k].unsqueeze(2), Sigma[k], self.sigma_sq[k]

        C, _c = self._C[k], self._c[k]

        S_inv = torch.inverse(S)
        E = S_inv + torch.inverse(s.view(-1, 1, 1)) * torch.transpose(C, -2, -1) @ C
        f = torch.solve(torch.transpose(C, -2, -1) @ (c.unsqueeze(2) - _c.unsqueeze(2)), s.view(-1, 1, 1) * E)[0]
        g = torch.solve(mu_, E @ S)[0]

        #B = torch.solve(I_l, E)[0]  # TODO: doubt it is more efficient

        #B = torch.inverse(E)

        mean = f + g
        #cov = B

        return mean.squeeze(-1), E

    def _get_cluster_mean_prec_z(self, c, mu, sigma):
        """
        Given a context c, it returns the parameters of the gaussian distrubiton p(z|c, k_i) for all i.
        :param c: vector of context (n_samples, context_dim)
        :param mu: vector of mu parameters
        :param sigma: vector of Sigma parameters
        :return:
        """
        means = []
        precs = []
        for k in range(self._n_cluster):
            mean, prec = self._get_mean_prec_z(c, torch.tensor([k]).repeat(c.shape[0]), mu, sigma)
            means.append(mean)
            precs.append(prec)
        return torch.stack(means).transpose(0, 1), torch.stack(precs).transpose(0, 1)

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

    def _test_log_policy(self, z, k, c, log_pi, mu, Sigma):
        z, k, c, log_pi, mu, Sigma = [torch.from_numpy(x) if type(x) is np.ndarray else x for x in
                                      [z, k, c, log_pi, mu, Sigma]]

        mean, prec = self._get_stable_mean_prec_z(c, k, mu, Sigma)
        log_z_given_k_c = multivariate_normal.MultivariateNormal(mean, precision_matrix=prec).log_prob(z)
        log_k_given_c = self._log_responsabilities(c, log_pi, mu, Sigma).T[range(c.shape[0]), k]

        return log_z_given_k_c + log_k_given_c

    def _log_policy(self, z, k, c, log_pi, mu, Sigma):
        z, k, c, log_pi, mu, Sigma = [torch.from_numpy(x) if type(x) is np.ndarray else x for x in
                                      [z, k, c, log_pi, mu, Sigma]]

        # mean, cov = self._get_mean_cov_z(c, k, mu, Sigma)
        # log_z_given_k_c = multivariate_normal.MultivariateNormal(mean, precision_matrix=torch.inverse(cov)).log_prob(z)
        # log_k_given_c = self._log_responsabilities(c, log_pi, mu, Sigma).T[range(c.shape[0]), k]
        #
        # return log_z_given_k_c + log_k_given_c
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

    def get_context_parameters(self, log_pi, mu, Sigma):
        ret_log_pi = []
        ret_mu = []
        ret_sigmas = []
        for i in range(self._n_cluster):
            k_i = torch.tensor([i])
            m, s = self.get_params_c_given_k(k_i, mu, Sigma)
            ret_mu.append(m)
            ret_sigmas.append(s)
            ret_log_pi.append(log_pi[k_i])
        return torch.stack(ret_log_pi).transpose(0, 1), torch.stack(ret_mu).transpose(0, 1), torch.stack(ret_sigmas).transpose(0, 1)

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

    def get_test_objective(self):
        log_p = self._test_log_policy(self._z, self._k, self._c_data,
                                 self.get_log_pi(self.log_base_pi), self.mu, self.get_Sigma(self._base_diag_Sigma))

        rho_num = log_p
        rho_den = torch.cat(self._rho_den)

        if self._normalize:
            w = torch.exp(rho_num - rho_den - sum_logs(rho_num - rho_den))
        else:
            w = torch.exp(rho_num - rho_den) / self._k.shape[0]

        ret = torch.sum(self._r*w)
        ret = self._get_gradient_from_torch(ret), ret.detach().numpy()

        return ret

    def get_objective(self):
        log_p = self._log_policy(self._z, self._k, self._c_data,
                                 self.get_log_pi(self.log_base_pi), self.mu, self.get_Sigma(self._base_diag_Sigma))

        rho_num = log_p
        rho_den = torch.cat(self._rho_den)

        if self._normalize:
            w = torch.exp(rho_num - rho_den - sum_logs(rho_num - rho_den))
        else:
            w = torch.exp(rho_num - rho_den) / self._k.shape[0]

        # todo: let's see how does it works
        ret = torch.sum(self._r*w) - 1E-4 * self.get_context_kl_tensor()
        ret = self._get_gradient_from_torch(ret), ret.detach().numpy()

        return ret

    def zero_grad(self):
        self._base_diag_Sigma.grad.zero_()
        self.mu.grad.zero_()
        self.log_base_pi.grad.zero_()

    def get_context_kl_tensor(self):
        log_pi_reg = self.get_log_pi(self.log_base_pi)
        mu_reg = self.mu
        sigma_reg = self.get_Sigma(self._base_diag_Sigma)

        log_pi_a, mu_a, sigma_a = self.get_context_parameters(log_pi_reg, mu_reg, sigma_reg)
        log_pi_b, mu_b, sigma_b = self.get_context_parameters(self._first_log_pi, self._first_mu, self._first_Sigma)
        kl, _ = self._get_kl(mu_b, torch.inverse(sigma_b), log_pi_b, mu_a, torch.inverse(sigma_a), log_pi_a)

        kl_mean = torch.mean(kl)
        return kl_mean

    def get_context_kl(self):
        """
        Add the forward KL entropy bound
        :param c: vector of context (n_samples x context_dim)
        :return:
        """

        kl_mean = torch.mean(self.get_context_kl_tensor())
        return self._get_gradient_from_torch(kl_mean), kl_mean.detach().numpy()

    def get_kl_stabilization(self):
        """
        Add the forward KL entropy bound
        :param c: vector of context (n_samples x context_dim)
        :return:
        """
        log_pi_reg = self.get_log_pi(self.log_base_pi).unsqueeze(0)
        mu_reg = self.mu.unsqueeze(0)
        sigma_reg = self.get_Sigma(self._base_diag_Sigma).unsqueeze(0)

        mu_a = self._first_mu.unsqueeze(0)
        lambda_a = torch.inverse(self._first_Sigma).unsqueeze(0)
        log_pi_a = self._first_log_pi.unsqueeze(0)

        kl, _ = self._get_kl(mu_reg, torch.inverse(sigma_reg), log_pi_reg, mu_a, lambda_a, log_pi_a)

        kl_mean = torch.mean(kl).detach().numpy()
        return self._get_gradient_from_torch(kl_mean), kl_mean.detach().numpy()

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

        mu_a, lambda_a = self._get_cluster_mean_prec_z(c, self._last_mu, self._last_Sigma)
        mu_b, lambda_b = self._get_cluster_mean_prec_z(c, mu_reg, sigma_reg)

        if self.__kl_type == "forward":
            kl, self.avg_entropy = self._get_kl(mu_b, lambda_b, log_pi_b.T, mu_a, lambda_a, log_pi_a.T)
        elif self.__kl_type == "reverse":
            kl, self.avg_entropy = self._get_kl(mu_a, lambda_a, log_pi_a.T, mu_b, lambda_b, log_pi_b.T)
        else:
            raise Exception("Value '%s' not known. Kl must be 'none', 'forward', 'reverse'." % self.__kl_type)

        kl_mean = torch.mean(kl)
        return self._get_gradient_from_torch(kl_mean), kl_mean.detach().numpy()

    def _get_kl(self, mu_a, lambda_a, log_pi_a, mu_b, lambda_b, log_pi_b):
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
        H = - self._get_parallel_cross_entropy(mu_a, lambda_a, log_pi_a, mu_a, lambda_a, log_pi_a)
        Cr_H = self._get_parallel_cross_entropy(mu_a, lambda_a, log_pi_a, mu_b, lambda_b, log_pi_b)
        return H + Cr_H, torch.mean(H).detach().numpy()

    def _get_parallel_cross_entropy(self, mu_a, lambda_a, log_pi_a, mu_b, lambda_b, log_pi_b):
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
            t_2 = -torch.log(torch.det(lambda_b[:, i]))  # log(det(S)) = log(det(S^-1)^-1) = -log(det(S^-1))
            prec_b = lambda_b[:, i]
            t_3 = torch.einsum("ijj->i", prec_b @ (torch.einsum("ij,ik->ijk", mu_a[:, i], mu_a[:, i]) + torch.inverse(lambda_a[:, i])))
            t_4 = torch.einsum("ij,ij->i", mu_b[:, i], ((prec_b @ mu_a[:, i].unsqueeze(2)).squeeze(-1)))
            t_5 = torch.einsum("ij,ij->i", mu_a[:, i], ((prec_b @ mu_b[:, i].unsqueeze(2)).squeeze(-1)))
            t_6 = torch.einsum("ij,ij->i", mu_b[:, i], ((prec_b @ mu_b[:, i].unsqueeze(2)).squeeze(-1)))
            cross_entropy_normals += torch.exp(log_pi_a[:, i]) * 0.5 * (t_1 + t_2 + t_3 - t_4 - t_5 + t_6)
        return cross_entropy_normals + cross_entropy_multinomials

    def get_f(self, x):
        # if self._x_f is not None:
        #     if np.array_equal(x, self._x_f):
        #         return -self._f
        self._set_x(x)
        # TODO; for testing, then remove
        _, self._h = self.get_context_kl()
        self.x_f = self._inner_x.copy()
        self._f_grad, self._f = self.get_objective()
        return -self._f

    def get_f_grad(self, x):
        # if self._x_f is not None:
        #     if np.array_equal(x, self._x_f):
        #         return -self._f_grad
        self._set_x(x)
        self.x_f = self._inner_x.copy()
        self._f_grad, self._f = self.get_objective()
        return -self._f_grad

    def get_g(self, x):
        # if self._x_g is not None:
        #     if np.array_equal(x, self._x_g):
        #         return -self._g + self._kl_bound
        self._set_x(x)
        self.x_g = self._inner_x.copy()
        self._g_grad, self._g = self.get_kl_regularization()
        return -self._g + self._kl_bound

    def get_g_grad(self, x):
        # if self._x_g is not None:
        #     if np.array_equal(x, self._x_g):
        #         return -self._g_grad
        self._set_x(x)
        self.x_g = self._inner_x.copy()
        self._g_grad, self._g = self.get_kl_regularization()
        return -self._g_grad

    def get_h(self, x):
        # if self._x_g is not None:
        #     if np.array_equal(x, self._x_g):
        #         return -self._g + self._kl_bound
        self._set_x(x)
        self.x_h = self._inner_x.copy()
        self._h_grad, self._h = self.get_context_kl()
        return -self._h + self._kl_bound_context

    def get_h_grad(self, x):
        # if self._x_g is not None:
        #     if np.array_equal(x, self._x_g):
        #         return -self._g_grad
        self._set_x(x)
        self.x_h = self._inner_x.copy()
        self._h_grad, self._h = self.get_context_kl()
        return -self._h_grad
