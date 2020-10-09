for i in 1 4
do
  for kl in 0.01 0.02 0.05 0.1 0.2 0.5 1. 2.
  do
    for fr in 0.5 0.6 0.7 0.8 0.9 1.
    do
    # python experiment_organizer.py --task_name reacher2d_${i} lampo_${i} -n 20 --batch_size 20 --n_evaluations 500 --imitation_learning 100 --kl_bound 0.05 --max_iter 10 --il_noise 0.3 -z
      python ct_experiment_organizer.py --task_name reacher2d_${i} ct_${i}_kl_${kl}_fr_${fr}_ndr -n 20 --batch_size 20 --n_evaluations 500 --imitation_learning 100 --kl_bound ${kl} --forgetting_rate ${fr} --max_iter 10 --il_noise 0.3 --not_dr
      python ct_experiment_organizer.py --task_name reacher2d_${i} ct_${i}_kl_${kl}_fr_${fr} -n 20 --batch_size 20 --n_evaluations 500 --imitation_learning 100 --kl_bound ${kl} --forgetting_rate ${fr} --max_iter 10 --il_noise 0.3
    done
  done
done

