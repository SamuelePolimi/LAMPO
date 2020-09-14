"""
Version of reinforce that includes

1. normalized importance sampling for data reuse
2. Reversed KL regularization in closed form

(DARE? Data Efficient Reinforce)
"""

from model import RLModel
from scipy import optimize
import numpy as np


class SafeReinforce:

    def __init__(self, rlmodel: RLModel):

        self.rlmodel = rlmodel

    def add_dataset(self, w, z, k, c, r):
        self.rlmodel.add_dataset(w, z, k, c, r)

    def improve(self, normalize=False):
        """
        Compute the safe gradient improvement direction and apply it.
        """

        ineq_cons = {'type': 'ineq',
                     'fun': self.rlmodel.get_g,
                     'jac': self.rlmodel.get_g_grad}

        xL = optimize.minimize(self.rlmodel.get_f, self.rlmodel._get_x(), constraints=[ineq_cons],
                               method='SLSQP', jac=self.rlmodel.get_f_grad, options={'ftol': 1e-9, 'disp': True})
        print("Sigma", np.exp(self.rlmodel._last_log_pi))
        print("Sigma", self.rlmodel._last_Sigma)
        print(xL)
        self.rlmodel.save_new_policy()