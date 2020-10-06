import numpy as np
from scipy.optimize import minimize


class EpisodicREPS:

    def __init__(self, kl_bound=0.05):
        self.kl_bound = kl_bound
        self._eta = np.array([1.])

    def optimize(self, rewards):
        R = np.array(rewards)
        f = lambda eta: self.kl_bound * eta**2 + eta**2 * np.log(np.mean(np.exp(R/eta**2)))
        self._eta = minimize(f, self._eta, method='Nelder-Mead', options={'xatol': 1e-8, 'disp': True}).x**2
        self._eta = self._eta[0]
        print("eta", self._eta)
        h = np.exp(rewards / (self._eta+1E-5))
        return h/np.mean(h)


