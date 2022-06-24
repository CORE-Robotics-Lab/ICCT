export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -u test.py \
  --env_name lunar \
  --seed 42 \
  --load_path log/ll \
  --num_episodes 5 \
  --load_file best_model \
  --gpu \
  | tee test.log
  
