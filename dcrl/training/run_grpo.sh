#!/usr/bin/env bash
set -euo pipefail

ulimit -n "$(ulimit -Hn)"
echo "Open-file limit: $(ulimit -Sn)"

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${PROJECT_ROOT}"
DCRL_ROOT=./dcrl
VERL_ROOT=${VERL_ROOT:-./verl}
if [[ ! -f "${VERL_ROOT}/verl/trainer/main_ppo.py" ]]; then
    echo "Missing local verl source; place dcrl/ and verl/ next to each other." >&2
    exit 1
fi

mkdir -p ./tmp/dcrl
# Temporary-directory APIs require an absolute runtime path, computed from this checkout.
export TMPDIR="$(pwd)/tmp/dcrl"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-./models/Qwen3-4B-Instruct-2507}
TRAIN_FILE=${TRAIN_FILE:-./data/train/train_ntrex_dcrl_k5_np.parquet}
VAL_FILE=${VAL_FILE:-./data/val/bouquet_dev.parquet}

SEED=${SEED:-42}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
PROJECT_NAME=${PROJECT_NAME:-dcrl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_a0_dcrl_dual_qe}

OUTPUT_DIR=${OUTPUT_DIR:-./outputs/${EXPERIMENT_NAME}}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/train_rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${OUTPUT_DIR}/validation_rollouts}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}/logs}

mkdir -p \
    "${ROLLOUT_DATA_DIR}" \
    "${VALIDATION_DATA_DIR}" \
    "${CHECKPOINT_DIR}" \
    "${LOG_DIR}"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TEST_FREQ=${TEST_FREQ:-50}
SAVE_FREQ=${SAVE_FREQ:-50}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.03}
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

export PYTHONPATH="$(pwd):$(cd "${VERL_ROOT}" && pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export REWARD_SERVER_URL=${REWARD_SERVER_URL:-http://localhost:8001}
export VALIDATION_REWARD_SERVER_URL=${VALIDATION_REWARD_SERVER_URL:-${REWARD_SERVER_URL}}

cd "${PROJECT_ROOT}"
"${PYTHON}" -m verl.trainer.main_ppo \
    --config-path "$(pwd)/dcrl/training" --config-name config \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.seed="${SEED}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.shuffle=False \
    data.truncation=error \
    data.continuous_token.enable=False \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.data_loader_seed="${SEED}" \
    actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.seed="${SEED}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.seed="${SEED}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=-1 \
    +actor_rollout_ref.rollout.repetition_penalty="${REPETITION_PENALTY}" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$((4 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    reward.reward_model.enable=False \
    reward.num_workers=8 \
    reward.custom_reward_function.path="${DCRL_ROOT}/reward/verl_reward.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=DCRLRewardManager \
    reward.reward_manager.module.path="${DCRL_ROOT}/reward/verl_reward.py" \
    trainer.use_v1=True \
    trainer.logger='["console","tensorboard","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    trainer.log_val_generations=20 \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.resume_mode=auto \
    "$@" 2>&1 | tee "${LOG_FILE}"
