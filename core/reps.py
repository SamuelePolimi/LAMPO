import numpy as np
from scipy.optimize import minimize

class EpisodicREPS:

    def __init__(self, kl_bound=0.5):
        self.kl_bound = kl_bound
        self._eta = np.array([10.])

    def optimize(self, rewards, parameters):
        R = np.array(rewards)
        f = lambda eta: self.kl_bound * eta**2 + eta**2 * np.log(np.mean(np.exp(R/eta**2)))
        self._eta = minimize(f, self._eta, method='Nelder-Mead', options={'xatol': 1e-8, 'disp': True}).x**2
        print("eta", self._eta)
        return np.exp(rewards / self._eta)


