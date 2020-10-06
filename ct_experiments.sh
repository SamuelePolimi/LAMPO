# Ablation study

# Imitation learning size

python ct_experiment_organizer.py il_size_1_dense -n 20 --n_evaluations 500 --imitation_learning 100 --batch_size 500 --max_iter 10 --dense_reward
python ct_experiment_organizer.py il_size_2_dense -n 20 --n_evaluations 500 --imitation_learning 200 --batch_size 500 --max_iter 10 --dense_reward
python ct_experiment_organizer.py il_size_3_dense -n 20 --n_evaluations 500 --imitation_learning 500 --batch_size 500 --max_iter 10 --dense_reward
python ct_experiment_organizer.py il_size_4_dense -n 20 --n_evaluations 500 --imitation_learning 1000 --batch_size 500 --max_iter 10 --dense_reward

# Reinforcement learning

python ct_experiment_organizer.py rl_size_1_dense -n 20 --n_evaluations 500 --imitation_learning 1000 --batch_size 20 --max_iter 10 --dense_reward
python ct_experiment_organizer.py rl_size_2_dense -n 20 --n_evaluations 500 --imitation_learning 1000 --batch_size 50 --max_iter 10 --dense_reward
python ct_experiment_organizer.py rl_size_3_dense -n 20 --n_evaluations 500 --imitation_learning 1000 --batch_size 100 --max_iter 10 --dense_reward
python ct_experiment_organizer.py rl_size_4_dense -n 20 --n_evaluations 500 --imitation_learning 1000 --batch_size 200 --max_iter 10 --dense_reward
python ct_experiment_organizer.py rl_size_5_dense -n 20 --n_evaluations 500 --imitation_learning 1000 --batch_size 500 --max_iter 10 --dense_reward

# Minimal

python ct_experiment_organizer.py small_dense -n 20 --n_evaluations 500 --imitation_learning 200 --batch_size 20 --max_iter 10 --dense_reward