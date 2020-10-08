"""
Version of reinforce that includes

1. normalized importance sampling for data reuse
2. Reversed KL regularization in closed form

(DARE? Data Efficient Reinforce)
"""

from core.model import RLModel
from scipy import optimize
import numpy as np
import time
import psutil


def displace(x, i, delta):
    ret = x.copy()
    ret[i] = ret[i] + delta
    return ret


def wait_resources():
    """
    If the cpus are too busy,it is not convenient to run simultaneously the process.
    :return:
    """
    while True:
        # A small random sleeping time, should make more probable that processes don't start exactly at the same time
        time.sleep(np.random.uniform())
        cpu = psutil.cpu_percent(interval=0.2)
        if cpu < 50.:
            break


class Lampo:

    def __init__(self, rlmodel: RLModel, wait=True):

        self.rlmodel = rlmodel
        self._wait = wait

    def add_dataset(self, w, z, k, c, r):
        self.rlmodel.add_dataset(w, z, k, c, r)

    def improve(self):
        """
        Compute the safe gradient improvement direction and apply it.
        """

        ineq_cons_1 = {'type': 'ineq',
                     'fun': self.rlmodel.get_g,
                     'jac': self.rlmodel.get_g_grad}

        ineq_cons_2 = {'type': 'ineq',
                     'fun': self.rlmodel.get_h,
                     'jac': self.rlmodel.get_h_grad}

        constraints = [ineq_cons_1, ineq_cons_2]
        if self.rlmodel._kl_bound_context < 0:
            constraints = [ineq_cons_1]

        if self._wait:
            wait_resources()

        xL = optimize.minimize(self.rlmodel.get_f, self.rlmodel._get_x(), constraints=constraints,
                               method='SLSQP', jac=self.rlmodel.get_f_grad, options={'ftol': 1e-9, 'disp': True})
        self.rlmodel._set_x(xL.x)
        # print(xL)
        self.rlmodel.save_new_policy()
        print("mu", self.rlmodel._last_mu)
        print("pi", np.exp(self.rlmodel._last_log_pi))
        print("Sigma", self.rlmodel._last_Sigma)
        print("context_kl", self.rlmodel.current_context_kl)

    def test_f_grad(self, x, delta=1E-5):
        f = self.rlmodel.get_f(x)
        grad = np.array([(self.rlmodel.get_f(displace(x, i, delta)) - f)/delta for i in range(x.shape[0])])
        return np.sqrt(np.mean(np.square(self.rlmodel.get_f_grad(x) - grad)))

    def test_g_grad(self, x, delta=1E-5):
        g = self.rlmodel.get_g(x)
        grad = np.array([(self.rlmodel.get_g(displace(x, i, delta)) - g)/delta for i in range(x.shape[0])])
        return np.sqrt(np.mean(np.square(self.rlmodel.get_g_grad(x) - grad)))

