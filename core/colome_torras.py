###################################################################################
# Algorithm implemented from Colome and Torras
###################################################################################
import numpy as np

from core.reps import EpisodicREPS
from sklearn.decomposition import PCA
from core.iterative_rwr_gmg import IRWRGMM


class CT_ImitationLearning:

    def __init__(self, state_dims, parameters_dims, latent_dims, n_clusters, use_dr=True):
        self._state_dims = state_dims
        self._latent_dims = latent_dims
        self._parameters_dims = parameters_dims
        self._n_clusters = n_clusters

        self._use_dr = use_dr

    def fit(self, context, parameters, forgetting_rate=0.5):
        if self._use_dr:
            self.pca = PCA(self._latent_dims)
            self.pca.fit(parameters)
            self._latent = self.pca.transform(parameters)
            new_data = np.concatenate([context, self._latent], axis=1)
        else:
            self._latent = parameters
            new_data = np.concatenate([context, parameters], axis=1)

        self._gmm = IRWRGMM(self._n_clusters, discount=forgetting_rate)
        self._gmm.fit_new_data(new_data, np.ones(context.shape[0]))

    def predict(self, context):
        z, k = self._gmm.predict(np.squeeze(context), dim=self._state_dims)
        if self._use_dr:
            return np.squeeze(self.pca.inverse_transform(np.expand_dims(z, 0))), z, k
        else:
            return z, z, k


class CT_ReinforcementLearning:

    def __init__(self, imitation, kl_bound=0.05):
        self._imitation = imitation
        self._reps = EpisodicREPS(kl_bound)

    def add_dataset(self, context, movement, reward):
        latent = self._imitation.pca.transform(movement)
        data = np.concatenate([context, latent], axis=1)
        w = self._reps.optimize(reward)
        self._imitation._gmm.fit_new_data(data, w)

    def generate_full(self, x, **kwargs):
        return self._imitation.predict(x)

