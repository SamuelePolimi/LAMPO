import argparse
from imitation_learning import PPCAImitation, RunModel
from core.plot import LampoMonitor
from core.lampo import Lampo
import numpy as np
from core.model import RLModel
from core.config import config


def get_arguments_dict():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_name", help="Where you would like to save the experimental results and configuration.")
    parser.add_argument("-t", "--task_name",
                        help="Task name.",
                        default="reach_target")
    parser.add_argument("-i", "--id",
                        help="Identifier of the process.",
                        type=int, default=10)
    parser.add_argument("-b", "--batch_size",
                        help="How many episodes before improvement.",
                        type=int, default=10)
    parser.add_argument("-l", "--imitation_learning",
                        help="How many episodes before improvement.",
                        type=int, default=10)
    parser.add_argument("-p", "--plot",
                        help="Show real time plots.",
                        action="store_true")
    parser.add_argument("-z", "--normalize",
                        help="Normalized Importance Sampling",
                        action="store_true")
    parser.add_argument("-s", "--save",
                        help="Save the results in the experiment directory.",
                        action="store_true")
    parser.add_argument("-c", "--context_kl_bound",
                        help="Bound the context kl.",
                        type=float,
                        default=50.)
    parser.add_argument("-k", "--kl_bound",
                        help="Bound the improvement kl.",
                        type=float,
                        default=0.2)
    parser.add_argument("-f", "--forward",
                        help="Bound the improvement kl.",
                        action="store_true")
    parser.add_argument("-m", "--max_iter",
                        help="Maximum number of iterations.",
                        type=int,
                        default=20)
    parser.add_argument("-e", "--n_evaluations",
                        help="Number of the evaluation batch.",
                        type=int,
                        default=500)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    x = np.load("prova.npz")
    args = get_arguments_dict()

    experiment_path = "experiments/%s/" % args.folder_name
    n_clusters = config[args.task_name]["n_cluster"]

    rlmodel = RLModel.load("prova.npz")
    run_model = RunModel(config[args.task_name]["task_class"], rlmodel, config[args.task_name]["n_features"], headless=False)

    s, r, actions, latent, cluster, observation = run_model.collect_samples(args.n_evaluations, isomorphic_noise=False)
    print("SUCCESS:", np.mean(s))
