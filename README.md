LAMPO: LAtent Movement Policy Optimization
==========================================

S. Tosatto, G. Chalvatzaki, J. Peters.

_In submission to RAL-ICRA 2021_

Install
-------
This python package currently depends on other packages which needs to be installed.
We recommend to create a python 3.6 virtual environment, for example using conda.

1. Install ROMI (robotic movement interface) which we use to learn ProMPS and as an interface to the real robot

```shell script
git clone https://github.com/SamuelePolimi/romi
cd romi
pip install -e . 
```

2. Install MPPCA (mixture of probabilistic component analyzers)

```shell script
cd ..
git clone https://github.com/SamuelePolimi/MPPCA
cd MPPCA/mppca/cmixture
python setup.py build_ext --inplace
cd ../..
pip install -e .
```

3. Install HeRL (Helper Reinforcement Learning), from which we use some specific dataset interface and saving modules.
```shell script
cd ..
git clone https://github.com/SamuelePolimi/MPPCA
cd MPPCA/mppca/cmixture
python setup.py build_ext --inplace
cd ../..
pip install -e .
```

You will require also `sklearn`, `pytorch`, `scipy`, `psutil` and `rlbench`.
To install `rlbench` follow carefully the instruction at `https://github.com/stepjam/RLBench`.

Run an experiment
-----------------

First of all, be aware that there is a configuration file which select some default hyperparameters which are also used in the paper.
The configuration file is present at `core/config.py` at it specity how many cluster to use and the latent space dimension.

All the other configurations are specified in the parameter list of `experiment_organizer.py` (for what regards experiments with LAMPO)
or `ct_experiment_organizer.py` to run the Colome and Torras algorithm referenced in the paper or the GMM+REPS (by calling the `--no_dr` option).

An example to run an experiment

```shell script
experiment_organizer.py folder_name --task_name reach_target --kl 0.2 -c -1 --n_runs 1 -l 200 --save -z --forward  --max_iter 10 --n_evaluations 500 --batch_size 100 --dense_reward 
```

We specify few of the options, but full description is enquirable by running `experiment_organizer.py ?`.

- `folder_name` the name of the folder where you want to save the data. it will be created in `experiments`.
- `--kl` the kl-bound
- `--forward` means that the kl will be forward (and not reverse), as mentioned in the paper
- `--n_runs` how many runs do you want to run?
- `--n_evaluations` at each iteration, how many evaluations do you want
- `--batch_size` how many sample should we take from the `n_evaluations` to learn
- `--imitation_learning` how many demonstration do we want to use

If you want, provided that the folder mentioned is created, you can launch also single experiments, very similarly

```shell script
experiment.py folder_name --task_name reach_target --kl 0.2 -c -1 -l 200 --save -z --forward  --max_iter 10 --n_evaluations 500 --batch_size 100 --dense_reward 
```

and you will also have the option

- `-p` to have a real-time plot
- `-v` to visualize the environment in Coppelia.

Similar running commands can be launched for `ct_experiment.py`, `ct_experiment_organizer.py`, `im_experiment.py` and `im_organizer.py` and `im_organize_ct.py` for only imitation learning.

Structure of the code:
----------------------

- `core` contains the important code
- `core/augmented_task` contains tasks from Coppelia and a bit refined (with dense reward and nicer interface)
- `core/collector.py` to collect experiments
- `core/colome_torras.py` the algorithm referenced with `CT` in the paper
- `core/config.py` configuration file
- `core/imitation_learning.py` perform imitation learning using MPPCA
- `core/iterative_rwr_gmm.py` implementation of reward weighted responsability Gaussian mixture model (used by Colome and Torras)
- `core/lab_connection.py` code to connect to the client of the real robot
- `core/lampo.py` the main optimization process run by our algorithm
- `core/model.py` the torch model of our probabilistic graph
- `core/reps.py` relative entropy policy searc, used by Colome and Torras
- `rl_bench_box.py` a boxing class for rl_bench. Here you find also `2d-reacher`
- `task_interface.py` a generic interface for tasks
- `tcp_protocol.py` a protocol for communicating with the client controlling the real robot

- `dataset_generator.py` generate a dataset of demonstration from RLBench and save it on disk
- `learn_movement.py` learn the movements of a dataset of demonstration and save it on disk
- `visualize_behavior.py` when you run an experiment, you will save some model file on the disk. You can run those model by specifying
the name of the folder, the name of the file (without `.npz`) and telling which task to run

Other files, like the `.sh` are scripts to launch experiments. 


For further inquiries 
---------------------

The code has been tested with python 3.6, and with all the dependencies installed should run. For any specific question at any regards,
please send an e-mail to `samuele.tosatto@gmail.com` or `samuele.tosatto@tu-darmstadt.de`.








TODO:
----
- [x] Imitation learning by conditioning a Mixture of Principal Component Analysis
- [x] Policy Improvement via Policy Gradient with KL constraint
- [x] Extraction of dense reward
- [x] More structured and clean version of the code
- [x] Parallel experiments
- [x] Inheritance to define dense reward
- [x] General interface working also with TCP/IP connection
- [x] Setting up experiments with the cluster (ask Joao)
- [x] Colome and Torras Baseline
- [x] Better analysis, tests and comments

