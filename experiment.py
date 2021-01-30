import argparse
from core.imitation_learning import PPCAImitation, RunModel
from mppca.mixture_ppca import MPPCA
from core.task_interface import TaskInterface
from core.plot import LampoMonitor
from core.lampo import Lampo
import numpy as np
from core.model import RLModel
from core.config import config
import json

import matplotlib.pyplot as plt


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
                        type=int, default=200)
    parser.add_argument("-p", "--plot",
                        help="Show real time plots.",
                        action="store_true")
    parser.add_argument("-v", "--visualize_robot",
                        help="Show robotic behavior",
                        action="store_true")
    parser.add_argument("-z", "--normalize",
                        help="Normalized Importance Sampling",
                        action="store_true")
    parser.add_argument("-s", "--save",
                        help="Save the results in the experiment directory.",
                        action="store_true")
    parser.add_argument("-d", "--load",
                        help="Load configuration from folder.",
                        action="store_true")
    parser.add_argument("-r", "--slurm",
                        help="Don't look for CPU usage.",
                        action="store_true")
    parser.add_argument("--il_noise",
                        help="Add noise on the context",
                        type=float,
                        default=0.03)
    parser.add_argument("--dense_reward",
                        help="Use dense reward",
                        action="store_true")
    parser.add_argument("-c", "--context_kl_bound",
                        help="Bound the context kl.",
                        type=float,
                        default=50.)
    parser.add_argument("-k", "--kl_bound",
                        help="Bound the improvement kl.",
                        type=float,
                        default=0.2)
    parser.add_argument("--context_reg",
                        help="Bound the improvement kl.",
                        type=float,
                        default=1E-4)
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
    parser.add_argument("--data_augment",
                        help="Number of artificially generated data (x times). (=1 means no data augmentation)",
                        type=int,
                        default=1)

    args = parser.parse_args()
    return args


class Objectview(object):

    def __init__(self, d):
        self.__dict__ = d


def process_parameters(parameters, n_samples, n_context, noise=0.03, augment=1):
    parameters = parameters[:n_samples].copy()
    data_list = []
    for i in range(augment):
        data_list.append(np.copy(parameters))
        data_list[-1][:, :n_context] += noise * np.random.normal(size=parameters[:, :n_context].shape)
    return np.concatenate(data_list, axis=0)


if __name__ == "__main__":
    args = get_arguments_dict()
    experiment_path = "experiments/%s/" % args.folder_name
    if args.load:
        with open(experiment_path + "configuration.json") as f:
            args = Objectview(json.load(f))

    n_clusters = config[args.task_name]["n_cluster"]

    task = config[args.task_name]["task_box"](not args.visualize_robot)     # type: TaskInterface

    state_dim = task.get_context_dim()


    parameters = task.get_demonstrations()[:args.imitation_learning]
    parameters = process_parameters(parameters, args.imitation_learning, state_dim, args.il_noise, augment=args.data_augment)

    mppca = MPPCA(n_clusters, config[args.task_name]["latent_dim"], n_init=500)
    mppca.fit(parameters)

    n_evaluation_samples = args.n_evaluations
    n_batch = args.batch_size

    kl_bound = args.kl_bound
    kl_context_bound = args.context_kl_bound
    if args.forward:
        kl_type = "forward"
    else:
        kl_type = "reverse"  # "reverse"

    normalize = args.normalize

    rl_model = RLModel(mppca, context_dim=config[args.task_name]["state_dim"], kl_bound=kl_bound,
                              kl_bound_context=kl_context_bound, kl_reg=args.context_reg,  normalize=normalize,
                              kl_type=kl_type)

    myplot = LampoMonitor(kl_bound, kl_context_bound=kl_context_bound,
                          title="class_log kl=%.2f, %d samples, kl_type=%s, normalize=%s" %
                          (kl_bound, n_batch, kl_type, normalize))

    sr = Lampo(rl_model, wait=not args.slurm)

    # TODO: start to modify here
    collector = RunModel(task, rl_model, args.dense_reward)
    for i in range(args.max_iter):

        s, r, actions, latent, cluster, observation = collector.collect_samples(n_evaluation_samples, isomorphic_noise=False)
        sr.add_dataset(actions[:n_batch], latent[:n_batch], cluster[:n_batch], observation[:n_batch], r[:n_batch])
        print("ITERATION", i)
        print("SUCCESS:", np.mean(s))

        myplot.notify_outer_loop(np.mean(s), np.mean(r))

        sr.improve()
        print("Optimization %f" % sr.rlmodel._f)
        print("KL %f <= %f" % (sr.rlmodel._g, kl_bound))
        if kl_context_bound> 0:
            print("KL context %f <= %f" % (sr.rlmodel._h, kl_context_bound))
        myplot.notify_inner_loop(sr.rlmodel._f, sr.rlmodel._g, sr.rlmodel.avg_entropy, sr.rlmodel._h)

        if args.plot:
            myplot.visualize()
        if args.save:
            myplot.save(experiment_path + "result_%d_%d.npz" % (args.id, i))
            sr.rlmodel.save(experiment_path + "model_%d_%d.npz" % (args.id, i))

    s, r, actions, latent, cluster, observation = collector.collect_samples(n_evaluation_samples, isomorphic_noise=False)

    print("ITERATION", args.max_iter)
    print("SUCCESS:", np.mean(s))
    myplot.notify_outer_loop(np.mean(s), np.mean(r))

    if args.plot:
        myplot.visualize(last=True)

    if args.save:
        myplot.save(experiment_path + "result_%d.npz" % args.id)
        sr.rlmodel.save(experiment_path + "model_%d.npz" % args.id)
