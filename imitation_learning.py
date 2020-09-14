from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig

import numpy as np
from dariaspy.trajectory import NamedTrajectory
from dariaspy.movement_primitives import MovementPrimitive, Group, ClassicSpace, \
    LearnTrajectory

from pyrep.backend import sim

from mppca.mixture_ppca import MPPCA


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

        space = ClassicSpace(group, n_features=n_features, regularization=1E-12)
        parameters = np.array([np.concatenate([s, np.array([l]), LearnTrajectory(space, traj).get_block_params()])
                               for s, l, traj in zip(states, lengths, trajectories)])
        np.save("parameters/%s_%d.npy" % (task.get_name(), n_features), parameters)
        env.shutdown()


class PPCAImitation:

    def __init__(self, task_class, state_dim, n_features, n_cluster, n_latent, headless=False, cov_reg=1E-8, n_samples=50):


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
        parameters = np.load("parameters/%s_%d.npy" % (self.task.get_name(), n_features))[:n_samples]
        # parameters = np.concatenate([parameters for _ in range(20)])
        # parameters[:, :3] += 0.03 * np.random.normal(size=parameters[:, :3].shape)
        self.mppca = MPPCA(n_cluster, n_latent, n_iterations=30, cov_reg=cov_reg)
        self.mppca.fit(parameters)

        mu = np.zeros((n_cluster, n_latent))
        sigma = np.array([np.eye(n_latent) for _ in range(n_cluster)])

        # self.rlmppca = RLMPPCA(self.mppca.log_pi, self.mppca.linear_transform,
        #                        self.mppca.means, self.mppca.covariances,
        #                        self.mppca.sigma_squared, mu, sigma, context_dim=state_dim
        #                        )
        self.rlmppca = None

        print(np.exp(self.mppca.log_pi))

        group = Group("rlbench", ["d%d" % i for i in range(7)] + ["gripper"])
        self.space = ClassicSpace(group, n_features=n_features)

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
            parameters.append(w)
            latent.append(z)
            cluster.append(k)
            mp = MovementPrimitive(self.space, MovementPrimitive.get_params_from_block(self.space, w[1:]))
            if self._headless:
                trajectory = mp.get_full_trajectory(duration=w[0], frequency=200)
            else:
                trajectory = mp.get_full_trajectory(duration=max(5*w[0], 0.1), frequency=200)
            tot_reward = 0.
            success = 0
            for a1 in trajectory.values:  # , a2 in zip(trajectory.values[:-1, :], trajectory.values[1:, :]):
                joint = a1  # (a2-a1)*20.
                joint_gripper = joint
                obs, reward, terminate = self.task.step(joint_gripper)
                tot_reward += reward
                state, points = sim.simCheckProximitySensor(self.task._task._initial_objs_in_scene[5][0]._handle,
                                                        self.task._robot.arm.get_tip().get_handle())
                points_1 = sim.simGetObjectPosition(self.task._task._initial_objs_in_scene[5][0]._handle,
                                                    self.task._robot.arm.get_tip().get_handle())

                tot_reward = - 5 * np.sqrt(points_1[0] ** 2 + points_1[1] ** 2 + points_1[2] ** 2)
                if state == 1 and terminate:
                    tot_reward = 1.
                    success = 1
                    break
                elif state == 1 and not terminate or state != 1 and terminate:
                    raise("There is something wrong")
                else:
                    pass

            # tot_reward =  -np.mean(np.abs(context - ob[1].gripper_pose[:3]))
            # tot_reward = -(w[0]-0.2)**2
            print(tot_reward)
            # print("my reward", -np.mean(np.abs(context - ob[1].gripper_pose[:3])))
            # print("parameters", parameters[-1])
            success_list.append(success)
            reward_list.append(tot_reward)
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