for kl in 0.5
do
  python experiment_organizer.py water_plants_${kl} --task_name water_plants -n 10 --n_evaluations 500 --imitation_learning 1000 --batch_size 500 -z --max_iter 7 -c -1 --il_noise 0.03  --kl_bound ${kl} --forward
done
