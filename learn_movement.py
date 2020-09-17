from imitation_learning import ImitationLearning
from core.config import config
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_name",
                        help="Task name.",
                        default="reach_target")
    args = parser.parse_args()

    return args


if __name__== "__main__":
    args = get_arguments()
    ImitationLearning(config[args.task_name]["task_class"], config[args.task_name]["n_features"], True, 200)