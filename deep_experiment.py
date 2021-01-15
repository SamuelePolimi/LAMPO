import argparse
import json
import torch

import gym
import rlbench.gym
import numpy as np
import stable_baselines3.common.env_checker
from gym.spaces import Box
from stable_baselines3 import PPO, SAC, HER

import matplotlib.pyplot as plt

from core.augmented_tasks.tasks import ReachTarget
from core.stats import StatsBox


def get_arguments_dict():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_name", help="Where you would like to save the experimental results and configuration.")
    parser.add_argument("-t", "--task_name",
                        help="Task name.",
                        default="reach_target-state-v0")
    parser.add_argument("-a", "--algorithm",
                        help="Algorithm Name.",
                        default="SAC")
    parser.add_argument("-r", "--slurm",
                        help="Don't look for CPU usage.",
                        action="store_true")
    parser.add_argument("-i", "--id",
                        help="Identifier of the process.",
                        type=int, default=5)

    args = parser.parse_args()
    return args.__dict__


def moving_average(m_list, length=100):
    signal = [np.mean(m_list[0:length])]*(length-1) + m_list
    m_list[0:length] = [np.mean(m_list[0:length])]*length
    ret = []
    for i in range(len(m_list)):
        ret.append(np.mean(signal[i:i+length]))
    return ret


algorithm = {"SAC": SAC,
             "HER": HER,
             "PPO": PPO
             }


if __name__ == "__main__":
    args = get_arguments_dict()
    env = StatsBox(gym.make(args["task_name"]), max_length=50, dense_reward=True, save_fr=100,
                   save_dest="deep_experiments/" + args["folder_name"] + "/%s_%s_%d" % (args["algorithm"], args["task_name"], args["id"]))
    print("Is Cuda Available", torch.cuda.is_available())
    print(stable_baselines3.common.env_checker.check_env(env))
    # exit()
    model = algorithm[args["algorithm"]]('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=2000)

    plt.title("SAC w Dense Rewards")
    plt.plot(env.returns, label="Returns")
    plt.plot(moving_average(env.returns), label="Returns - Moving Average")
    plt.ylabel("Dense Return")
    plt.xlabel("Episodes")
    plt.legend(loc="best")
    plt.savefig("returns1.pdf")
    plt.show()

    plt.title("SAC w Dense Rewards")
    plt.plot(env.successes)
    plt.plot(env.successes, label="Successes")
    plt.plot(moving_average(env.successes), label="Successes - Moving Average")
    plt.ylabel("Dense Return")
    plt.xlabel("Episodes")
    plt.legend(loc="best")
    plt.savefig("returns2.pdf")
    plt.show()