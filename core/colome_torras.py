###################################################################################
# Algorithm implemented from Colome and Torras
###################################################################################
import numpy as np

from herl.dataset import Dataset, Domain, Variable
from core.reps import EpisodicREPS
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from core.iterative_rwr_gmg import IRWRGMM


class CT_ImitationLearning:

    def __init__(self, state_dims, parameters_dims, latent_dims, n_clusters):
        self._state_dims = state_dims
        self._latent_dims = latent_dims
        self._parameters_dims = parameters_dims
        self._n_clusters = n_clusters
        self._domain = Domain(Variable("context", state_dims), Variable("latent", latent_dims))

    def fit(self, context, parameters):
        self.pca = PCA(self._latent_dims)
        self.pca.fit(parameters)
        self._latent = self.pca.transform(parameters)
        new_data = np.concatenate([context, self._latent], axis=1)
        self._gmm = IRWRGMM(self._n_clusters)
        self._gmm.fit_new_data(new_data, np.ones(context.shape[0]))

    def predict(self, context):
        z, k = self._gmm.predict(np.squeeze(context), dim=self._state_dims)
        return self.pca.inverse_transform(z), z, k


class CT_ReinforcementLearning:

    def __init__(self, imitation):
        self._imitation = imitation
        self._reps = EpisodicREPS()

    def add_dataset(self, context, movement, reward):
        latent = self._imitation.pca.transform(movement)
        data = np.concatenate([context, latent], axis=0)
        w = self._reps.optimize(reward, data)
        self._imitation.fit_new_data(data, w)

    def generate_full(self, x, **kwargs):
        self._imitation.predict(x)

