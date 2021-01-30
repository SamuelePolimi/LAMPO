from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig

import numpy as np
from romi.trajectory import NamedTrajectory
from romi.movement_primitives import MovementPrimitive, Group, ClassicSpace, \
    LearnTrajectory
from mppca.multiprocess import MultiProcess

from pyrep.backend import sim

from mppca.mixture_ppca import MPPCA
from core.model import RLModel
from core.task_interface import TaskInterface

import matplotlib.pyplot as plt


class ImitationLearning:

    def __init__(self, task_class, n_features, load, n_movements):
        """
        Learn the Movement from demonstration.

        :param task_class: Task that we aim to learn
        :param n_features: Number of RBF in n_features
        :param load: Load from data
        :param n_movements: how many movements do we want to learn
        """

        frequency = 200

        # To use 'saved' demos, set the path below, and set live_demos=False
        live_demos = not load
        DATASET = '' if live_demos else 'datasets'

        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)
        self._task_class = task_class
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)

        group = Group("rlbench", ["d%d" % i for i in range(7)] + ["gripper"])

        env = Environment(
            action_mode, DATASET, obs_config, headless=True)
        env.launch()

        task = env.get_task(task_class)

        trajectories = []
        states = []

        lengths = []

        print("Start Demo")
        demos = task.get_demos(n_movements, live_demos=live_demos)
        print("End Demo")

        init = True
        for demo in demos:
            trajectory = NamedTrajectory(*group.refs)
            t = 0
            for ob in demo:
                if t == 0:
                    if init:
                        print("State dim: %d" % ob.task_low_dim_state.shape[0])
                        init = False
                    states.append(ob.task_low_dim_state)
                kwargs = {"d%d" % i: ob.joint_positions[i] for i in range(ob.joint_positions.shape[0])}
                kwargs["gripper"] = ob.gripper_open
                trajectory.notify(duration=1/frequency,
                                  **kwargs)
                t += 1
            lengths.append(t/200.)
            trajectories.append(trajectory)

        space = ClassicSpace(group, n_features=n_features, regularization=1E-15)
        z = np.linspace(-0.2, 1.2, 1000)
        Phi = space.get_phi(z)
        for i in range(n_features):
            plt.plot(z, Phi[:, i])
        plt.show()
        parameters = np.array([np.concatenate([s, np.array([l]), LearnTrajectory(space, traj).get_block_params()])
                               for s, l, traj in zip(states, lengths, trajectories)])
        np.save("parameters/%s_%d.npy" % (task.get_name(), n_features), parameters)
        env.shutdown()


class PPCAImitation:

    def __init__(self, task_class, state_dim, n_features, n_cluster, n_latent, parameters=None, headless=False,
                 cov_reg=1E-8, n_samples=50, dense_reward=False, imitation_noise=0.03):


        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)
        self._obs_config = obs_config

        self._state_dim = state_dim
        self._headless = headless

        action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
        self._task_class = task_class
        self._action_mode = action_mode
        self.env = Environment(
            action_mode, "", obs_config, headless=headless)
        self.env.launch()

        self.task = self.env.get_task(task_class)
        if parameters is None:
            self.parameters = np.load("parameters/%s_%d.npy" % (self.task.get_name(), n_features))[:n_samples]
        # parameters = np.concatenate([parameters for _ in range(20)])
        self.imitation_noise = imitation_noise
        self.parameters[:, :3] += imitation_noise * np.random.normal(size=self.parameters[:, :3].shape)
        self.mppca = MPPCA(n_cluster, n_latent, n_iterations=30, cov_reg=cov_reg, n_init=500)
        self.mppca.fit(self.parameters)

        self.rlmppca = None
        self.dense_reward = dense_reward

        print(np.exp(self.mppca.log_pi))

        group = Group("rlbench", ["d%d" % i for i in range(7)] + ["gripper"])
        self.space = ClassicSpace(group, n_features=n_features)

        print("mpcca learned")

    def stop(self, joint_gripper, previous_reward):
        if previous_reward == 0.:
            success = 0.
            for _ in range(50):
                obs, reward, terminate = self.task.step(joint_gripper)
                if reward == 1.0:
                    success = 1.
                    break
            return self.task._task.get_dense_reward(), success
        return self.task._task.get_dense_reward(), 1.

    def collect_samples(self, n_episodes=50, noise=True, isomorphic_noise=False):

        success_list = []
        reward_list = []
        latent = []
        cluster = []
        observations = []
        parameters = []
        ob = self.task.reset()
        for i in range(n_episodes):
            context = ob[1].task_low_dim_state
            observations.append(context)
            w, z, k = self.rlmppca.generate_full(np.expand_dims(context, 0), noise=noise, isomorphic_noise=isomorphic_noise)
            # w = self.parameters[i, 3:]
            parameters.append(w)
            latent.append(z)
            cluster.append(k)
            mp = MovementPrimitive(self.space, MovementPrimitive.get_params_from_block(self.space, w[1:]))
            duration = 1 if w[0] < 0 else w[0]
            print(duration)
            if self._headless:
                trajectory = mp.get_full_trajectory(duration=duration, frequency=200)
            else:
                trajectory = mp.get_full_trajectory(duration=5*duration, frequency=200)
            tot_reward = 0.
            success = 0
            for a1 in trajectory.values:  # , a2 in zip(trajectory.values[:-1, :], trajectory.values[1:, :]):
                joint = a1  # (a2-a1)*20.
                joint_gripper = joint
                obs, reward, terminate = self.task.step(joint_gripper)
                if reward == 1. or terminate==1.:
                    if reward == 1.:
                        success = 1.
                    break
            tot_reward, success = self.stop(joint_gripper, success)
            # tot_reward =  -np.mean(np.abs(context - ob[1].gripper_pose[:3]))
            # tot_reward = -(w[0]-0.2)**2
            print(tot_reward)
            # print("my reward", -np.mean(np.abs(context - ob[1].gripper_pose[:3])))
            # print("parameters", parameters[-1])
            success_list.append(success)

            if self.dense_reward:
                reward_list.append(tot_reward)
            else:
                reward_list.append(success)

            ob = self.task.reset()

        print("-"*50)
        print("Total reward", np.mean(reward_list))
        print("-"*50)
        return np.array(success_list), np.array(reward_list), np.array(parameters), np.array(latent), np.array(cluster), np.array(observations)

    def run_episode(self):
        self.env = Environment(
            self._action_mode, "", self._obs_config, headless=self._headless)
        self.env.launch()

        self.task = self.env.get_task(self._task_class)

    def shutdown(self):
        self.env.shutdown()


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
            print("RECEIVED CONTEXT", np.expand_dims(context, 0))
            w, z, k = self._rl_model.generate_full(np.expand_dims(context, 0), noise=noise, isomorphic_noise=isomorphic_noise)

            parameters.append(w)
            latent.append(z)
            cluster.append(k)

            success, tot_reward = self.task.send_movement(w[1:], w[0])
            print(success, tot_reward, k, z)
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
