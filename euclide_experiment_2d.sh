for i in 1 2 3 4
do
  python experiment_organizer.py --task_name reacher2d_${i} lampo_${i} -n 20 --batch_size 20 --n_evaluations 500 --imitation_learning 100 --kl_bound 0.05 --max_iter 10 --il_noise 0.3 -z
  python ct_experiment_organizer.py --task_name reacher2d_${i} ct_${i} -n 20 --batch_size 20 --n_evaluations 500 --imitation_learning 100 --kl_bound 0.05 --max_iter 10 --il_noise 0.3 -z
done

