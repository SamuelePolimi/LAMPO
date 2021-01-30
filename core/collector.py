import numpy as np

from core.task_interface import TaskInterface
from core.model import RLModel


class RunModel:

    def __init__(self, task: TaskInterface, rl_model: RLModel, dense_reward=False):
        self.task = task
        self._rl_model = rl_model
        self.dense_reward = dense_reward

    def collect_samples(self, n_episodes=50, noise=True, isomorphic_noise=False):

        success_list = []
        reward_list = []
        latent = []
        cluster = []
        observations = []
        parameters = []

        for i in range(n_episodes):
            self.task.reset()
            context = self.task.read_context()
            observations.append(context)
            w, z, k = self._rl_model.generate_full(np.expand_dims(context, 0), noise=noise, isomorphic_noise=isomorphic_noise)

            parameters.append(w)
            latent.append(z)
            cluster.append(k)

            success, tot_reward = self.task.send_movement(w[1:], w[0])
            print(success, tot_reward)
            success_list.append(success)

            if self.dense_reward:
                reward_list.append(tot_reward)
            else:
                reward_list.append(success)

        print("-"*50)
        print("Total reward", np.mean(reward_list))
        print("-"*50)
        return np.array(success_list), np.array(reward_list), np.array(parameters), np.array(latent),\
               np.array(cluster), np.array(observations)