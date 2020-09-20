# Ablation study

# Imitation learning size

python experiment_organizer.py il_size_1 -n 10 --n_evaluations 500 --imitation_learning 100 --batch_size 500 -z --max_iter 10 --dense_reward
python experiment_organizer.py il_size_2 -n 10 --n_evaluations 500 --imitation_learning 200 --batch_size 500 -z --max_iter 10 --dense_reward
python experiment_organizer.py il_size_3 -n 10 --n_evaluations 500 --imitation_learning 500 --batch_size 500 -z --max_iter 10 --dense_reward
python experiment_organizer.py il_size_4 -n 10 --n_evaluations 500 --imitation_learning 1000 --batch_size 500 -z --max_iter 10 --dense_reward

# Reinforcement learning

python experiment_organizer.py rl_size_1 -n 10 --n_evaluations 500 --imitation_learning 1000 --batch_size 20 -z --max_iter 10 --dense_reward
python experiment_organizer.py rl_size_2 -n 10 --n_evaluations 500 --imitation_learning 1000 --batch_size 50 -z --max_iter 10 --dense_reward
python experiment_organizer.py rl_size_3 -n 10 --n_evaluations 500 --imitation_learning 1000 --batch_size 100 -z --max_iter 10 --dense_reward
python experiment_organizer.py rl_size_4 -n 10 --n_evaluations 500 --imitation_learning 1000 --batch_size 200 -z --max_iter 10 --dense_reward
python experiment_organizer.py rl_size_5 -n 10 --n_evaluations 500 --imitation_learning 1000 --batch_size 500 -z --max_iter 10 --dense_reward

# Minimal

python experiment_organizer.py small -n 10 --n_evaluations 500 --imitation_learning 200 --batch_size 20 -z --max_iter 10 --dense_reward
