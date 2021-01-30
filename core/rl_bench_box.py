"""
Implement a common interface for the RLBench which works also for the TCP-IP connection.
"""
import numpy as np
import matplotlib.patches as patches
import time

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig

from core.task_interface import TaskInterface
from core.rrt_star import RRTStar
from romi.movement_primitives import ClassicSpace, MovementPrimitive, LearnTrajectory
from romi.groups import Group
from romi.trajectory import NamedTrajectory, LoadTrajectory


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
            trajectory = mp.get_full_trajectory(duration=min(duration, 1), frequency=200)
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


class Forward2DKinematics:

    def __init__(self, d1, d2):
        self._d1 = d1
        self._d2 = d2

    def _link(self, d):
        return np.array([d, 0.])

    def _rot(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    def get_forward(self, theta1, theta2):
        x1 = self._rot(theta1) @ self._link(self._d1)
        r1 = self._rot(theta1) @ self._rot(0.)
        r2 = self._rot(theta2) @ r1
        x2 = r2 @ self._link(self._d2) + x1
        return x2

    def get_full_forward(self, theta1, theta2):
        x1 = self._rot(theta1) @ self._link(self._d1)
        r1 = self._rot(theta1) @ self._rot(0.)
        r2 = self._rot(theta2) @ r1
        x2 = r2 @ self._link(self._d2) + x1
        return x1, x2

    def get_loss(self, theta1, theta2, goal):
        ref = self.get_forward(theta1, theta2)
        return np.mean((ref - goal)**2)

    def jac(self, theta1, theta2, goal, delta=1E-5):
        ref = self.get_loss(theta1, theta2, goal)
        j1 = (self.get_loss(theta1 + delta, theta2, goal) - ref)/delta
        j2 = (self.get_loss(theta1, theta2+delta, goal) - ref)/delta
        return np.array([j1, j2])

    def get_trajectory(self, theta1, theta2, goal, v=0.1):
        conf = [np.array([theta1, theta2])]
        for _ in range(200):
            conf.append(conf[-1]-v*self.jac(conf[-1][0], conf[-1][1], goal))
        return conf, [self.get_forward(c[0], c[1]) for c in  conf]


class Reacher2D(TaskInterface):

    def __init__(self,  n_features, points=0, headless=True):

        super().__init__(n_features)
        self._group = Group("reacher2d", ["j%d" % i for i in range(2)])
        self._space = ClassicSpace(self._group, n_features)

        self._state_dim = 2
        self._headless = headless

        self._n_points = points
        self._goals = [self._point(3/2, np.pi/8),
                       self._point(1., np.pi/2 + np.pi/8),
                       self._point(2/3, np.pi + np.pi/4),
                       self._point(1/2, 3/2*np.pi + np.pi/6)]
        self._kinematics = Forward2DKinematics(1., 1.)

        self._context = None

    def _point(self, d, theta):
        return d*np.array([np.cos(theta), np.sin(theta)])

    def _generate_context(self, goal=None):
        if self._n_points == 0:
            d = np.random.uniform(0, 1)
            a = np.random.uniform(-np.pi, np.pi)
            return self._point(d, a)
        else:
            if goal is None:
                k = np.random.choice(range(self._n_points))
            else:
                k = goal
            g = self._goals[k]
            d = np.random.uniform(0, 1/5)
            a = np.random.uniform(-np.pi, np.pi)
            return g + self._point(d, a)

    def give_example(self, goal=None):
        goal = self._generate_context(goal)
        conf, traj = self._kinematics.get_trajectory(0., 0., goal)
        return goal, conf, traj

    def _generate_demo(self):
        goal = self._generate_context()
        conf, traj = self._kinematics.get_trajectory(0., 0., goal)
        trajectory = NamedTrajectory(*self._group.refs)
        for c in conf:
            trajectory.notify(duration=1/100.,
                              j0=c[0], j1=c[1])
        return goal, np.array([3.]), LearnTrajectory(self._space, trajectory).get_block_params()

    def get_context_dim(self):
        return self._state_dim

    def read_context(self):
        return self._context

    def get_demonstrations(self):
        return np.array([np.concatenate(self._generate_demo(), axis=0) for _ in range(100)])

    def send_movement(self, weights, duration):
        mp = MovementPrimitive(self._space, MovementPrimitive.get_params_from_block(self._space, weights))
        duration = 1 if duration < 0 else duration
        trajectory = mp.get_full_trajectory(duration=duration, frequency=200)

        vals = trajectory.get_dict_values()

        reward = -self._kinematics.get_loss(vals["j0"][-1], vals["j1"][-1], self._context)

        return reward, reward

    def reset(self):
        self._context = self._generate_context()


class ObstacleRectangle:

    def __init__(self, x, y, dx, dy):
        self.x1 = x
        self.x2 = x + dx
        self.y1 = y
        self.y2 = y + dy
        self._patch = patches.Rectangle((x, y), dx, dy,
                                        linewidth=1, edgecolor='r', facecolor='r')

    def check_collision_point(self, point):
        if self.x1 <= point[0] <= self.x2:
            if self.y1 <= point[1] <= self.y2:
                return True
        return False

    def check_collision_points(self, points):
        for point in points:
            if self.check_collision_point(point):
                return True
        return False

    def draw(self, ax):
        ax.add_patch(self._patch)


def _positive_range(angle):
    ret = angle
    while ret < 0.:
        ret += 2 * np.pi
    while ret > 2 * np.pi:
        ret -= 2 * np.pi
    return ret


def _2p_range(angle):
    ret = angle
    while ret < -np.pi:
        ret += 2 * np.pi
    while ret > np.pi:
        ret -= 2 * np.pi
    return ret


def get_angle_between(angle_1, angle_2):
    _angle_1 = _positive_range(angle_1)
    _angle_2 = _positive_range(angle_2)
    if np.abs(_angle_1 - _angle_2) < np.pi:
        return np.abs(_angle_1 - _angle_2)
    else:
        return 2*np.pi - np.abs(_angle_1 - _angle_2)


def get_mid_angle(angle_1, angle_2, length):
    _angle_1 = _positive_range(angle_1)
    _angle_2 = _positive_range(angle_2)
    if np.abs(_angle_1 - _angle_2) < np.pi:
        ret = _angle_1 + np.clip(_angle_2 - _angle_1, -length, length)
    else:
        if _angle_2 > _angle_1:
            delta = get_angle_between(_angle_2, _angle_1)
            delta = min(delta, length)
            ret = _angle_1 - delta
        else:
            delta = get_angle_between(_angle_2, _angle_1)
            delta = min(delta, length)
            ret = _angle_1 + delta
    ret = _2p_range(ret)
    return ret


def sampling():
    return np.random.uniform(-np.pi * np.ones(2), np.pi * np.ones(2))
# print(get_angle_between(-np.pi+0.1, np.pi-0.1))


class ObstacleReacher2d(TaskInterface):

    def __init__(self,  n_features, headless=True):

        super().__init__(n_features)
        self._group = Group("reacher2d", ["j%d" % i for i in range(2)])
        self._space = ClassicSpace(self._group, n_features)

        self._state_dim = 2
        self._headless = headless

        self._obstacle = ObstacleRectangle(0.5, 0.5, 0.25, 0.25)

        self._kinematics = Forward2DKinematics(1., 1.)

        self._rrt_star = RRTStar(np.array([0., 0.]), self.close_to_goal, sampling, 0.05,
                                 self.get_configuration_distance,
                                 collision_detector=self.check_collision,
                                 goal_distance=self.distance_to_goal,
                                 get_mid_point=self.get_mid_configuration,
                                 star=True,
                                 star_distance=0.1)

        self._context = None
        self.reset()

    def get_configuration_distance(self, conf_1, conf_2):
        d1 = get_angle_between(conf_1[0], conf_2[0])
        d2 = get_angle_between(conf_1[1], conf_2[1])
        return np.sqrt(d1**2 + d2**2)

    def get_mid_configuration(self, conf_1, conf_2, length=0.1):
        d1 = get_mid_angle(conf_1[0], conf_2[0], length)
        d2 = get_mid_angle(conf_1[1], conf_2[1], length)
        return np.array([d1, d2])

    def check_collision(self, configuration):
        x1, x2 = self._kinematics.get_full_forward(configuration[0], configuration[1])
        points_l1 = np.linspace(np.zeros_like(x1), x1)
        if self._obstacle.check_collision_points(points_l1):
            # print("Collision conf", configuration[0], configuration[1])
            # print("Collision points", x1, x2)
            return True
        points_l2 = np.linspace(x1, x2)
        if self._obstacle.check_collision_points(points_l2):
            # print("Collision conf", configuration[0], configuration[1])
            # print("Collision points", x1, x2)
            return True
        return False

    def distance_to_goal(self, configuration):
        x = self._kinematics.get_forward(configuration[0], configuration[1])
        return np.sqrt(np.sum((x - self._context)**2))

    def close_to_goal(self, configuration):
        x = self._kinematics.get_forward(configuration[0], configuration[1])
        return np.sqrt(np.sum((x - self._context)**2)) < 0.1

    def close_to_goal_env(self, configuration):
        # print("Goal was is", self._context)
        # print("Configuration is", configuration)
        # print("Reached position is", self._kinematics.get_forward(configuration[0], configuration[1]))
        x = self._kinematics.get_forward(configuration[0], configuration[1])
        return np.sqrt(np.sum((x - self._context)**2)) < 0.4

    def draw(self, configuration, ax):
        x1, x2 = self._kinematics.get_full_forward(configuration[0], configuration[1])
        ax.plot([0, x1[0]], [0, x1[1]], c='gray')
        ax.plot([x2[0], x1[0]], [x2[1], x1[1]], c='gray')

    def _point(self, d, theta):
        return d*np.array([np.cos(theta), np.sin(theta)])

    def _generate_context(self, goal=None):
        d = np.random.uniform(0, 1.8)
        a = np.random.uniform(-np.pi, np.pi)
        ret = self._point(d, a)
        while self._obstacle.check_collision_point(ret):
            d = np.random.uniform(0, 1.8)
            a = np.random.uniform(-np.pi, np.pi)
            ret = self._point(d, a)
        return ret

    def give_example(self, goal=None):
        # TODO: change
        goal = self._generate_context(goal)
        conf, traj = self._kinematics.get_trajectory(0., 0., goal)
        return goal, conf, traj

    def _generate_demo(self, reuse_rrt_graph=True):
        self.reset()
        goal = self.read_context()

        graph = self._rrt_star.graph if reuse_rrt_graph else None

        self._rrt_star = RRTStar(np.array([0., 0.]), self.close_to_goal, sampling, 0.05,
                self.get_configuration_distance,
                collision_detector=self.check_collision,
                goal_distance=self.distance_to_goal,
                get_mid_point=self.get_mid_configuration,
                star=True,
                graph=graph,
                star_distance=0.1)

        if len(self._rrt_star.graph.all_nodes) == 0:
            for _ in range(5000):
                self._rrt_star.add_point()

        self._rrt_star.evaluate()

        if self._rrt_star.is_goal_reached():
            print("RRT SUCCESS")
        else:
            print("RRT FAIL")

        last_node = self._rrt_star.closest_node
        traj = []
        for node in last_node.get_path_to_origin():
            pos = node.position
            traj.append(pos)

        trajectory = NamedTrajectory(*self._group.refs)
        for c in traj:
            trajectory.notify(duration=1/100.,
                              j0=c[0], j1=c[1])
        return goal, np.array([len(traj)/100.]), trajectory


    def get_context_dim(self):
        return self._state_dim

    def read_context(self):
        return self._context

    def save_demonstration(self):
        demos = [] #np.array([np.concatenate(self._generate_demo(), axis=0) for _ in range(5)])
        for i in range(100000):
            # current_demos = np.array([np.concatenate(self._generate_demo(), axis=0) for _ in range(5)])
            start = time.time()
            if i % 20 == 0:
                goal, duration, trajectory = self._generate_demo(reuse_rrt_graph=False)
            else:
                goal, duration, trajectory = self._generate_demo(reuse_rrt_graph=True)
            params = LearnTrajectory(self._space, trajectory).get_block_params()
            current_demo = np.concatenate([goal, duration, params], axis=0)
            trajectory.save("trajectories/trajectory_%d.npy" % i)
            demos.append(current_demo)

            print("demo-time: %f" % (time.time() - start))
            if i % 5 == 0:
                np.save("obstacle.npy", demos)
        demo = np.load("core/demonstrations/reacher2d_obstacle.npy")
        print("Loaded demo", demo.shape)
        return demo

    def get_demonstrations(self):
        demo = np.load("core/demonstrations/reacher2d_obstacle.npy")
        print("Loaded demo", demo.shape)
        return demo

    def get_success_demo(self):
        demos = self.get_demonstrations()
        ret = []
        for i, demo in enumerate(demos):
            self._context = demo[:2]
            traj = LoadTrajectory("trajectories/trajectory_%d.npy" % i)
            traj_val = traj.get_dict_values()
            conf = np.array([traj_val["j0"][-1], traj_val["j1"][-1]])
            ret.append(self.close_to_goal(conf))

        return np.sum(ret)/len(ret)

    def send_movement(self, weights, duration):
        mp = MovementPrimitive(self._space, MovementPrimitive.get_params_from_block(self._space, weights))
        duration = 1 if duration < 0 else duration
        trajectory = mp.get_full_trajectory(duration=duration, frequency=50)

        vals = trajectory.get_dict_values()

        for v0, v1 in zip(vals["j0"], vals["j1"]):
            if self.check_collision(np.array([v0, v1])):
                return False, -1.

        reward = -self._kinematics.get_loss(vals["j0"][-1], vals["j1"][-1], self._context)

        return self.close_to_goal_env(np.array([vals["j0"][-1], vals["j1"][-1]])), reward

    def reset(self):
        self._context = self._generate_context()