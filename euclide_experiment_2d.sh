for i in 1 2 3 4
do
  python experiment_organizer.py --task_name reacher2d_${i}_0.3 lampo_${i} -n 20 --batch_size 20 --n_evaluations 500 --imitation_learning 100 --kl_bound 0.5 --max_iter 5 --il_noise 0.3 -z
  python ct_experiment_organizer.py --task_name reacher2d_${i}_0.3 ct_${i} -n 20 --batch_size 20 --n_evaluations 500 --imitation_learning 100 --kl_bound 0.5 --max_iter 5 --il_noise 0.3
  python experiment_organizer.py --task_name reacher2d_${i}_0.15 lampo_${i} -n 20 --batch_size 20 --n_evaluations 500 --imitation_learning 100 --kl_bound 0.5 --max_iter 5 --il_noise 0.15 -z
  python ct_experiment_organizer.py --task_name reacher2d_${i}_0.15 ct_${i} -n 20 --batch_size 20 --n_evaluations 500 --imitation_learning 100 --kl_bound 0.5 --max_iter 5 --il_noise 0.15
done

