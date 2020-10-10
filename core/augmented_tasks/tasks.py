from rlbench.tasks import ReachTarget as RT, WaterPlants, CloseDrawer as CD
from pyrep.backend import sim
import numpy as np
from rlbench.backend.task import Task
import math


class AugmentedTask(Task):

    def set_additional_info(self, *args, **kwargs):
        raise Exception("not implemented")

    def get_dense_reward(self) -> float:
        raise Exception("not implemented")


class ReachTarget(RT):

    def set_additional_info(self, robot):
        self._robot = robot

    def get_dense_reward(self):
        state, points = sim.simCheckProximitySensor(self._initial_objs_in_scene[5][0]._handle,
                                                    self.robot.arm.get_tip().get_handle())
        points_1 = sim.simGetObjectPosition(self._initial_objs_in_scene[5][0]._handle,
                                            self.robot.arm.get_tip().get_handle())

        reward = - np.sqrt(points_1[0] ** 2 + points_1[1] ** 2 + points_1[2] ** 2)
        if state == 1:
            reward = 1.

        return reward


class CloseDrawer(CD):

    def set_additional_info(self, robot):
        self._robot = robot

    def get_dense_reward(self):
        rew = np.abs(self.joints[0].get_joint_position() - self._success_conditions[0]._original_pos)
        if rew >= 0.06:
            return 1.
        else:
            return rew/0.12
