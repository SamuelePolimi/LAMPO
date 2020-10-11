for kl in 0.5
do
  python experiment_organizer.py close_drawer_${kl} --task_name close_drawer -n 5 --n_evaluations 500 --imitation_learning 200 --batch_size 100 -z --max_iter 5 -c -1 --il_noise 0.03  --kl_bound ${kl} --forward
done
