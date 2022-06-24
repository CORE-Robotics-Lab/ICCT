export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=1

python -u train.py \
  --env_name lunar \
  --policy_type ddt \
  --seed 0 \
  --num_leaves 8 \
  --lr 5e-4 \
  --ddt_lr 5e-4 \
  --buffer_size 1000000 \
  --batch_size 256 \
  --gamma 0.99 \
  --tau 0.01 \
  --learning_starts 10000 \
  --eval_freq 1500 \
  --min_reward 225 \
  --training_steps 500000 \
  --log_interval 4 \
  --save_path log/ll \
  --use_individual_alpha \
  --submodels \
  --hard_node \
  --gpu \
  --argmax_tau 1.0 \
  --sparse_submodel_type 2 \
  --num_sub_features 2 \
  | tee train_ll.log