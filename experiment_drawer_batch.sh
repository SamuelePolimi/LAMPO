for kl in 0.1 0.2 0.5 1.0 2.0:
do
  python experiment_organizer.py close_drawer_${kl} --task_name close_drawer -n 5 --n_evaluations 500 --imitation_learning 200 --batch_size 100 -z --max_iter 10 -c -1 --id 0 --il_noise 0.0 --context_reg -1.0  --kl_bound ${kl}
done
