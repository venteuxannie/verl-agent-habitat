set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
# export CUDA_VISIBLE_DEVICES=0,1,2,3,7
export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

# train_data_size=16
# val_data_size=16
# group_size=8
train_data_size=8
val_data_size=8
group_size=4

# Model output storage configuration
OUTPUT_DIR="./rollouts_logs/grpo-habitat-new_resolution"
ROLLOUT_OUTPUT_DIR="$OUTPUT_DIR/rollouts"
VALIDATION_OUTPUT_DIR="$OUTPUT_DIR/validations"

# Storage frequency configuration
ROLLOUT_LOG_FREQ=1  # Log rollout data every N steps (set to 1 for every step)

# Create output directories
mkdir -p $ROLLOUT_OUTPUT_DIR
mkdir -p $VALIDATION_OUTPUT_DIR

echo "Model output directories created:"
echo "  Rollout outputs: $ROLLOUT_OUTPUT_DIR (every $ROLLOUT_LOG_FREQ steps)"
echo "  Validation outputs: $VALIDATION_OUTPUT_DIR (every 10 steps)"

python3 -m examples.data_preprocess.prepare \
--mode 'visual' \
--train_data_size $train_data_size \
--val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files=$HOME/data/verl-agent/visual/train.parquet \
data.val_files=$HOME/data/verl-agent/visual/test.parquet \
data.train_batch_size=$train_data_size \
data.val_batch_size=$val_data_size \
data.max_prompt_length=1024 \
data.max_response_length=512 \
data.filter_overlong_prompts=True \
data.truncation='error' \
data.image_key=images \
data.return_raw_chat=True \
actor_rollout_ref.model.path=/data1/tct_data/LLaMA-Factory/output/Qwen2-VL-2B-replica10-vg-sft-new_resolution \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.01 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
actor_rollout_ref.rollout.name=$ENGINE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.enable_chunked_prefill=False \
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=False \
actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.use_invalid_action_penalty=True \
actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
algorithm.use_kl_in_reward=False \
env.env_name=habitat \
env.dataset_name=ReplicaCAD \
env.seed=0 \
env.max_steps=10 \
env.rollout.n=$group_size \
trainer.critic_warmup=0 \
trainer.logger=['console','wandb'] \
trainer.project_name='verl_agent' \
trainer.experiment_name='sft-grpo-new_resolution-bs_32-4cards' \
trainer.n_gpus_per_node=4 \
trainer.nnodes=1 \
trainer.save_freq=25 \
trainer.resume_mode=auto \
trainer.test_freq=10 \
trainer.total_epochs=200 \
trainer.val_before_train=True \
trainer.rollout_data_dir=$ROLLOUT_OUTPUT_DIR \
trainer.validation_data_dir=$VALIDATION_OUTPUT_DIR \
trainer.rollout_log_freq=$ROLLOUT_LOG_FREQ \
trainer.log_val_generations=5 $@

echo "Training completed!"
echo "Model outputs stored in:"
echo "  Rollout data: $ROLLOUT_OUTPUT_DIR"
echo "  Validation data: $VALIDATION_OUTPUT_DIR"
echo ""
echo "To analyze the stored model outputs, you can use:"
echo "  ls -la $ROLLOUT_OUTPUT_DIR/"
echo "  head -n 1 $ROLLOUT_OUTPUT_DIR/*.jsonl | jq ."
echo "  python analyze_grpo_outputs.py --output-dir $OUTPUT_DIR"