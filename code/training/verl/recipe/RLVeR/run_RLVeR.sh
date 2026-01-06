
set -x

#export WANDB_API_KEY=YOUR_WANDB_API_KEY
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
echo "Here we are..."

MODEL_PATH="Qwen/Qwen2.5-3B"
TRAIN="<PATH_TO_TRAIN_DATA>"
VAL="<PATH_TO_VAL_DATA>"

REWARD="/path/to/batch_reward_python_sandbox_with_noise.py"

FALSE_POSITIVE_RATE=0.0
FALSE_NEGATIVE_RATE=0.0
KL_COEF_REWARD=0.0
KL_COEF_ACTOR=0.0
# Calculate B as a number
B=$(echo "1 - $FALSE_POSITIVE_RATE - $FALSE_NEGATIVE_RATE" | bc -l)
# Format B to 2 decimal places for the experiment name
B_FORMATTED=$(printf "%.2f" $B)
python3 path/to/verl/verl/trainer/main_ppo.py \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.train_batch_size=16 \
    data.max_prompt_length=4000 \
    data.max_response_length=4000 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    reward_model.reward_manager='batch' \
    custom_reward_function.path="$REWARD" \
    custom_reward_function.name="compute_score_batch" \
    +custom_reward_function.reward_kwargs.batch_size=1024 \
    +custom_reward_function.reward_kwargs.sim_tool_test="sandbox" \
    +custom_reward_function.reward_kwargs.sim_tool_train="sandbox" \
    +custom_reward_function.reward_kwargs.false_positive_rate=$FALSE_POSITIVE_RATE \
    +custom_reward_function.reward_kwargs.false_negative_rate=$FALSE_NEGATIVE_RATE \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n=8 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=16 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=$KL_COEF_REWARD \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF_ACTOR \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name='multi-arm-bandit' \
    trainer.experiment_name="noise_B${B_FORMATTED}_FP${FALSE_POSITIVE_RATE}_FN${FALSE_NEGATIVE_RATE}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=1000 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    +trainer.save_last=True 2>&1 | tee verl_demo_$(date +"%Y%m%d_%H%M%S").log $@
