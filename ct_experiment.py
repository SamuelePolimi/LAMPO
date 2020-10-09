from core.config import config
import argparse
from core.collector import RunModel
from core.colome_torras import CT_ImitationLearning, CT_ReinforcementLearning
from core.task_interface import TaskInterface
from core.plot import LampoMonitor
import numpy as np
import json


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
    parser.add_argument("--not_dr",
                        help="Don't do dimensionality reduction.",
                        action="store_true")
    parser.add_argument("--forgetting_rate",
                        help="The forgetting rate of the IRWR-GMM.",
                        type=float,
                        default=1.)

    args = parser.parse_args()
    return args


class Objectview(object):

    def __init__(self, d):
        self.__dict__ = d


def process_parameters(parameters, n_samples, n_context, noise=0.03):
    parameters = parameters[:n_samples].copy()
    parameters[:, :n_context] += noise * np.random.normal(size=parameters[:, :n_context].shape)
    return parameters


if __name__ == "__main__":
    print("ciao")
    args = get_arguments_dict()
    experiment_path = "ct_experiments/%s/" % args.folder_name
    if args.load:
        with open(experiment_path + "configuration.json") as f:
            args = Objectview(json.load(f))

    n_clusters = config[args.task_name]["n_cluster"]

    task = config[args.task_name]["task_box"](not args.visualize_robot)     # type: TaskInterface

    state_dim = task.get_context_dim()


    parameters = task.get_demonstrations()
    parameters = process_parameters(parameters, args.imitation_learning, state_dim, args.il_noise)

    imitation = CT_ImitationLearning(state_dim, parameters.shape[1] - config[args.task_name]["latent_dim"],
                                     config[args.task_name]["latent_dim"], n_clusters, use_dr=not args.not_dr)
    imitation.fit(parameters[:, :state_dim], parameters[:, state_dim:], forgetting_rate=args.forgetting_rate)

    n_evaluation_samples = args.n_evaluations
    n_batch = args.batch_size

    kl_bound = args.kl_bound
    if args.forward:
        kl_type = "forward"
    else:
        kl_type = "reverse"  # "reverse"

    rl_model = CT_ReinforcementLearning(imitation, kl_bound=kl_bound)

    myplot = LampoMonitor(kl_bound, kl_context_bound=0.,
                          title="class_log kl=%.2f, %d samples" %
                          (kl_bound, n_batch))

    collector = RunModel(task, rl_model, args.dense_reward)
    for i in range(args.max_iter):

        s, r, actions, latent, cluster, observation = collector.collect_samples(n_evaluation_samples, isomorphic_noise=False)
        rl_model.add_dataset(observation[:n_batch], actions[:n_batch], r[:n_batch])
        print("ITERATION", i)
        print("SUCCESS:", np.mean(s))

        myplot.notify_outer_loop(np.mean(s), np.mean(r))

        # sr.improve()
        # print("Optimization %f" % sr.rlmodel._f)
        # print("KL %f <= %f" % (sr.rlmodel._g, kl_bound))
        # if kl_context_bound> 0:
        #     print("KL context %f <= %f" % (sr.rlmodel._h, kl_context_bound))
        myplot.notify_inner_loop(0., 0., 0., 0.)

        if args.plot:
            myplot.visualize()

    s, r, actions, latent, cluster, observation = collector.collect_samples(n_evaluation_samples, isomorphic_noise=False)

    print("ITERATION", args.max_iter)
    print("SUCCESS:", np.mean(s))
    myplot.notify_outer_loop(np.mean(s), np.mean(r))

    if args.plot:
        myplot.visualize(last=True)

    if args.save:
        myplot.save(experiment_path + "result_%d.npz" % args.id)

