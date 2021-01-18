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
                        default="reach_target-state-v0")
    parser.add_argument("-a", "--algorithm",
                        help="Algorithm Name.",
                        default="SAC")
    parser.add_argument("-n", "--n_runs",
                        help="How many runs would you like to perform.",
                        type=int, default=10)
    parser.add_argument("-d", "--dense",
                        help="Dense reward.",
                        action="store_true")
    parser.add_argument("-s", "--timesteps",
                        help="Number of Timesteps.",
                        type=int, default=500000)
    parser.add_argument("-r", "--slurm",
                        help="Don't look for CPU usage.",
                        action="store_true")
    parser.add_argument("--id_start",
                        help="How many runs would you like to perform.",
                        type=int, default=-1)

    args = parser.parse_args()

    return args.__dict__


def work(arg_parse, id):
    proc = subprocess.Popen(["python", "deep_experiment.py"] + experiment_line(arg_parse, id))
    proc.wait()


def experiment_line(arg_parse: dict, id):
    positional = ["folder_name"]
    booleans = ["slurm", "dense"]
    exclude = ["n_runs", "id_start", "date"]
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
    return ret


if __name__ == "__main__":

    args_dict = get_arguments_dict()
    now = datetime.now()  # current date and time
    args_dict["date"] = now.strftime("%m/%d/%Y, %H:%M:%S")

    print("-"*20)
    print(" Experiment '%s' started." % args_dict["folder_name"])
    print("-"*20)

    experiment_path = "deep_experiments/%s/" % args_dict["folder_name"]
    pprint.pprint(args_dict)
    print(experiment_line(args_dict, 2))

    if args_dict["id_start"] < 0:
        if create_folder(experiment_path):
            with open(experiment_path + 'configuration.json', 'w') as fp:
                json.dump(args_dict, fp, indent=4, sort_keys=True)

                if not args_dict["slurm"]:

                    tp = ThreadPool(1)
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
