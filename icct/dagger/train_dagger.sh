export OMP_NUM_THREADS=1

python -u train_dagger.py \
  --env_name lunar \
  --max_depth 5 \
  --n_rollouts 10 \
  --iterations 100 \
  --max_samples 1000000 \
  --eval_episodes 50 \
  --oracle_load_path oracle_models \
  --oracle_load_file ll_mlp_max \
  --save saved_dt_models/ll \
  --seed 5 \
  | tee train.log