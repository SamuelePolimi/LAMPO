# python experiment_organizer.py obstacle_2d --task_name reacher2d_obstacle -n 5 --n_evaluations 500 --imitation_learning 200 --batch_size 100 -z --max_iter 5 -c -1 --il_noise 0.03  --kl_bound 0.15 --forward
# python experiment_organizer.py obstacle_2d_test_def --task_name reacher2d_obstacle -n 10 --n_evaluations 300 --imitation_learning 9000 --batch_size 300 -z --max_iter 10 -c 10. --il_noise 0.03  --dense  --kl_bound 0.1 --forward
rm -r experiments/obstacle_2d_test_def_more_1
python experiment_organizer.py obstacle_2d_test_def_more_1 --task_name reacher2d_obstacle -n 1 --n_evaluations 300 --imitation_learning 16500 --batch_size 300 -z --max_iter 10 -c 10. --il_noise 0.001  --dense  --kl_bound 0.1 --forward --start_id 1
