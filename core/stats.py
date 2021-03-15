from gym import Env
import numpy as np
from herl.dict_serializable import DictSerializable

from core.augmented_tasks.tasks import ReachTarget, CloseDrawer


class StatsBox(Env, DictSerializable):

    load_fn = DictSerializable.get_numpy_load()
    metadata = {'render.modes': ['human']}

    def __init__(self, env, max_length=np.inf, dense_reward=True, save_fr=10, save_dest="state_box", render=False):
        Env.__init__(self)
        DictSerializable.__init__(self, DictSerializable.get_numpy_save())
        self.eval_env = env
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        if env is not None:
            self.action_space = self.eval_env.action_space
            # Example for using image as input:
            self.observation_space = self.eval_env.observation_space
        self._dense_reward = dense_reward
        self.partial_reward = 0.
        self.partial_length = 0
        self.returns = []
        self.episode_lengths = []
        self.successes = []
        self._unused = True
        self._max_length = max_length
        self.max_episode_steps = max_length
        self._save_fr = save_fr
        self._save_dest = save_dest
        self._render = render

    @staticmethod
    def load_from_dict(**kwargs):
        ret = StatsBox(kwargs["env"], kwargs["max_length"], kwargs["dense_reward"], kwargs["save_fr"],
                       kwargs["save_dest"])
        ret.partial_reward = kwargs["partial_reward"]
        ret.partial_length = kwargs["partial_length"]
        ret.returns = kwargs["returns"].tolist()
        ret.episode_lengths = kwargs["episode_lengths"].tolist()
        ret.successes = kwargs["successes"].tolist()
        ret._unused = kwargs["unused"]
        return ret

    @staticmethod
    def load(file_name: str, env):
        """

        :param file_name:
        :param domain:
        :return:
        """
        file = StatsBox.load_fn(file_name)
        return StatsBox.load_from_dict(env=env, **file)

    def _get_dict(self):
        return {
            "max_length": self._max_length,
            "dense_reward": self._dense_reward,
            "save_fr": self._save_fr,
            "save_dest": self._save_dest,
            "partial_reward": self.partial_reward,
            "partial_length": self.partial_length,
            "returns": self.returns,
            "episode_lengths": self.episode_lengths,
            "successes": self.successes,
            "unused": self._unused
        }

    def step(self, action):
        self._unused = False
        self.partial_length += 1
        s, r, d, i = self.eval_env.step(action)
        # if 'render_mode' in self.eval_env.spec.kwargs.keys():
        #     if self.eval_env.spec.kwargs['render_mode'] == 'human':
        #         self.eval_env.render()
        if self.partial_length >= self._max_length:
            d = True
        if d:
            if r == 1:
                self.successes.append(1)
            else:
                self.successes.append(0)
            if self._dense_reward:
                if self.eval_env.spec._env_name=="reach_target-state":
                    r = ReachTarget.get_dense_reward(self.eval_env.task._task)
                elif self.eval_env.spec._env_name=="close_drawer-state":
                    r = CloseDrawer.get_dense_reward(self.eval_env.task._task)
                else:
                    raise Exception("No dense reward for the selected task.")

            self.partial_reward += r
            self.returns.append(self.partial_reward)
            self.episode_lengths.append(self.partial_length)
            print("return", self.partial_reward)
            print("length", self.partial_length)
            self.partial_reward = 0.
            self.partial_length = 0
            self._unused = True
        else:
            r = 0.
        if self._render:
            self.render()
        return s, r, d, i

    def reset(self):
        if self._save_fr is not None:
            if len(self.returns) % self._save_fr == 0:
                DictSerializable.save(self, self._save_dest)

        if not self._unused:
            if self.partial_reward == 1:
                self.successes.append(1)
            else:
                self.successes.append(0)
            self.returns.append(self.partial_reward)
            self.episode_lengths.append(self.partial_length)
            print("return", self.partial_reward)
            print("length", self.partial_length)
            self.partial_reward = 0.
            self.partial_length = 0
            return self.eval_env.reset()  # reward, done, info can't be included
        else:
            self.partial_reward = 0.
            return self.eval_env.reset()

    def render(self, mode='human'):
        return self.eval_env.render(mode)

    def close(self):
        return self.eval_env.close()