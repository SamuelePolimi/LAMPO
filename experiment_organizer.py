import argparse
import json
from datetime import datetime
import subprocess
from multiprocessing.pool import ThreadPool
import os
import pprint
from core.fancy_print import f_print, PTYPE


def create_folder(path):
    try:
        os.makedirs(path)
    except OSError:
        f_print("Creation of the directory %s failed." % path, PTYPE.warning)
        return False
    else:
        f_print("Successfully created the directory %s." % path, PTYPE.ok_green)
        return True


def get_arguments_dict():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_name", help="Where you would like to save the experimental results and configuration.")
    parser.add_argument("-t", "--task_name",
                        help="Task name.",
                        default="reach_target")
    parser.add_argument("-n", "--n_runs",
                        help="How many runs would you like to perform.",
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
    parser.add_argument("-s", "--save",
                        help="Save the results in the experiment directory.",
                        action="store_true")
    parser.add_argument("-d", "--load",
                        help="Load configuration from folder.",
                        action="store_true")
    parser.add_argument("-r", "--slurm",
                        help="Don't look for CPU usage & immediately run experiments.",
                        action="store_true")
    parser.add_argument("-z", "--normalize",
                        help="Self-Normalize Importance Sampling.",
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
    parser.add_argument("--il_noise",
                        help="Add noise on the context",
                        type=float,
                        default=0.03)
    parser.add_argument("--dense_reward",
                        help="Use dense reward",
                        action="store_true")
    parser.add_argument("--id_start",
                        help="How many runs would you like to perform.",
                        type=int, default=-1)
    parser.add_argument("--data_augment",
                        help="Augment the data x times (if x=1 data is not augmented)",
                        type=int, default=1)

    args = parser.parse_args()

    return args.__dict__


def work(arg_parse, id):
    proc = subprocess.Popen(["python", "experiment.py"] + experiment_line(arg_parse, id))
    proc.wait()


def experiment_line(arg_parse: dict, id):
    positional = ["folder_name"]
    booleans = ["plot", "normalize", "forward", "load", "dense_reward", "slurm"]
    exclude = ["n_runs", "date", "save", "id_start"]
    ret = []
    for p in positional:
        ret.append(arg_parse[p])
    for k, v in arg_parse.items():
        if k not in positional + exclude:
            k_real = "--" + k
            if k not in booleans:
                ret.append(k_real)
                ret.append(str(v))
            else:
                if v:
                    ret.append(k_real)

    ret.append("--id")
    ret.append(str(id))
    ret.append("-s")
    return ret


if __name__ == "__main__":

    args_dict = get_arguments_dict()
    now = datetime.now()  # current date and time
    args_dict["date"] = now.strftime("%m/%d/%Y, %H:%M:%S")

    print("-"*20)
    print(" Experiment '%s' started." % args_dict["folder_name"])
    print("-"*20)

    experiment_path = "experiments/%s/" % args_dict["folder_name"]
    pprint.pprint(args_dict)
    print(experiment_line(args_dict, 2))

    if args_dict["id_start"] < 0:
        if create_folder(experiment_path):
            with open(experiment_path + 'configuration.json', 'w') as fp:
                json.dump(args_dict, fp, indent=4, sort_keys=True)

            if not args_dict["slurm"]:

                tp = ThreadPool(5)
                for idx in range(args_dict["n_runs"]):
                    tp.apply_async(work, (args_dict, idx))

                tp.close()
                tp.join()
    else:
        tp = ThreadPool(5)
        for idx in range(args_dict["id_start"], args_dict["n_runs"] + args_dict["id_start"]):
            tp.apply_async(work, (args_dict, idx))

        tp.close()
        tp.join()
