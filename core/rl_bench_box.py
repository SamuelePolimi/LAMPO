"""
Implement a common interface for the RLBench which works also for the TCP-IP connection.
"""
import numpy as np

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig

from core.task_interface import TaskInterface
from romi.movement_primitives import ClassicSpace, MovementPrimitive
from romi.groups import Group


class RLBenchBox(TaskInterface):

    def __init__(self, task_class, state_dim, n_features, headless=True):

        super().__init__(n_features)
        self._group = Group("rlbench", ["d%d" % i for i in range(7)] + ["gripper"])
        self._space = ClassicSpace(self._group, n_features)

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
        self._obs = None

    def get_context_dim(self):
        return self._state_dim

    def read_context(self):
        return self._obs[1].task_low_dim_state

    def get_demonstrations(self):
        file = "parameters/%s_%d.npy" % (self.task.get_name(), self._space.n_features)
        try:
            return np.load(file)
        except:
            raise Exception("File %s not found. Please consider running 'dataset_generator.py'" % file)

    def _stop(self, joint_gripper, previous_reward):
        if previous_reward == 0.:
            success = 0.
            for _ in range(50):
                obs, reward, terminate = self.task.step(joint_gripper)
                if reward == 1.0:
                    success = 1.
                    break
            return self.task._task.get_dense_reward(), success
        return self.task._task.get_dense_reward(), 1.

    def send_movement(self, weights, duration):
        mp = MovementPrimitive(self._space, MovementPrimitive.get_params_from_block(self._space, weights))
        duration = 1 if duration < 0 else duration
        if self._headless:
            trajectory = mp.get_full_trajectory(duration=duration, frequency=200)
        else:
            trajectory = mp.get_full_trajectory(duration=5 * duration, frequency=200)
        tot_reward = 0.
        success = 0
        for a1 in trajectory.values:  # , a2 in zip(trajectory.values[:-1, :], trajectory.values[1:, :]):
            joint = a1  # (a2-a1)*20.
            joint_gripper = joint
            obs, reward, terminate = self.task.step(joint_gripper)
            if reward == 1. or terminate == 1.:
                if reward == 1.:
                    success = 1.
                break
        tot_reward, success = self._stop(joint_gripper, success)
        return success, tot_reward

    def reset(self):
        self._obs = self.task.reset()

