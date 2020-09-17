import argparse
from imitation_learning import PPCAImitation
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
                        type=int, default=200)
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
    args = get_arguments_dict()

    experiment_path = "experiments/%s/" % args.folder_name
    n_clusters = config[args.task_name]["n_cluster"]

    rl_ppca = PPCAImitation(config[args.task_name]["task_class"],
                            config[args.task_name]["state_dim"],
                            config[args.task_name]["n_features"],
                            n_clusters,
                            config[args.task_name]["latent_dim"],
                            headless=True,
                            n_samples=args.imitation_learning,
                            cov_reg=1E-8)

    n_evaluation_samples = args.n_evaluations
    n_batch = args.batch_size

    kl_bound = args.kl_bound
    kl_context_bound = args.context_kl_bound
    if args.forward:
        kl_type = "forward"
    else:
        kl_type = "reverse"  # "reverse"

    normalize = args.normalize
    rl_ppca.rlmppca = RLModel(rl_ppca.mppca, context_dim=config[args.task_name]["state_dim"], kl_bound=kl_bound,
                              kl_bound_stab=kl_context_bound, normalize=normalize,
                              kl_type=kl_type)

    myplot = LampoMonitor(kl_bound, "class_log kl=%.2f, %d samples, kl_type=%s, normalize=%s" %
                          (kl_bound, n_batch, kl_type, normalize))
    sr = Lampo(rl_ppca.rlmppca)

    for i in range(args.max_iter):

        s, r, actions, latent, cluster, observation = rl_ppca.collect_samples(n_evaluation_samples, isomorphic_noise=False)
        sr.add_dataset(actions[:n_batch], latent[:n_batch], cluster[:n_batch], observation[:n_batch], r[:n_batch])
        print("ITERATION", i)
        print("SUCCESS:", np.mean(s))
        myplot.notify_outer_loop(np.mean(s), np.mean(r))

        sr.improve()
        print("Optimization %f" % sr.rlmodel._f)
        print("KL %f <= %f" % (sr.rlmodel._g, kl_bound))
        print("KL context %f <= %f" % (sr.rlmodel._h, kl_context_bound))
        myplot.notify_inner_loop(sr.rlmodel._f, sr.rlmodel._g, sr.rlmodel.avg_entropy, sr.rlmodel._h)

        if args.plot:
            myplot.visualize()

    s, r, actions, latent, cluster, observation = rl_ppca.collect_samples(n_evaluation_samples, isomorphic_noise=False)

    print("ITERATION", args.max_iter)
    print("SUCCESS:", np.mean(s))
    myplot.notify_outer_loop(np.mean(s), np.mean(r))

    if args.plot:
        myplot.visualize(last=True)

    if args.save:
        myplot.save(experiment_path + "result_%d.npz" % args.id)
        sr.rlmodel.save(experiment_path + "model_%d.npz" % args.id)

