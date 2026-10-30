# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def dcrl_token_terms(teacher_logits, student_logits, labels, old_log_probs, advantages,
                          temperature, clip_ratio, alpha, beta):
    """Conditional full-vocabulary KL at teacher-sampled prefixes (not sequence KL).

    Each KL has a different stop-gradient side. The old teacher denominator is
    fixed for the entire rollout batch. No KL is inserted into advantages.
    """
    teacher = torch.log_softmax(teacher_logits.float() / temperature, dim=-1)
    student = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_lp = teacher.gather(-1, labels[:, None]).squeeze(-1)
    student_lp = student.gather(-1, labels[:, None]).squeeze(-1)
    old = old_log_probs.detach().float()
    advantage = advantages.detach().float()
    # Guard floating-point overflow only; this is separate from PPO clipping.
    t_ratio = (teacher_lp - old).clamp(-20, 20).exp()
    s_ratio = (student_lp - old).clamp(-20, 20).exp()

    def policy_loss(ratio):
        return -torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)

    t_pg, s_pg = policy_loss(t_ratio), policy_loss(s_ratio)
    t_kl = (teacher.exp() * (teacher - student.detach())).sum(-1)
    s_kl = (teacher.detach().exp() * (teacher.detach() - student)).sum(-1)
    total = alpha * (t_pg + beta * t_kl) + (1 - alpha) * (s_pg + beta * s_kl)
    # All returned diagnostics are detached; only total participates in backward.
    diagnostics = torch.stack([
        t_pg, s_pg, t_kl, s_kl, t_ratio, s_ratio,
        ((t_ratio - 1).abs() > clip_ratio).float(),
        ((s_ratio - 1).abs() > clip_ratio).float(),
        -(teacher.exp() * teacher).sum(-1),
        -(student.exp() * student).sum(-1),
        ((teacher_lp - old).abs() > 20).float(),
        ((student_lp - old).abs() > 20).float(),
    ], dim=-1).detach()
    return total, diagnostics


def dcrl_loss(config, model_output, data, dp_group=None, dcrl_config=None):
    """Sequence-mean/token-mean joint surrogate, normalized over active rows globally."""
    from torch.utils.checkpoint import checkpoint

    names = ["teacher_pg_loss", "student_pg_loss", "teacher_kl", "student_kl",
             "teacher_ratio", "student_ratio", "teacher_clip_fraction", "student_clip_fraction",
             "teacher_entropy", "student_entropy", "teacher_log_ratio_guard", "student_log_ratio_guard"]
    alpha, beta = float(dcrl_config.alpha), float(dcrl_config.kl_coef)
    chunk_size = int(dcrl_config.kl_chunk_size)
    temperature = float(tu.get_non_tensor_data(data, "temperature", 1.0))
    scale = float(data["dp_size"]) / max(float(data["dcrl_active_count"]), 1.0)
    # Keep zero-gradient graphs on inactive DP ranks so FSDP collectives still match.
    loss = model_output["teacher_logits"][0][:1, :1].sum() * 0
    loss = loss + model_output["student_logits"][0][:1, :1].sum() * 0
    diagnostics = loss.detach().new_zeros(len(names))
    for i, (t_logits, s_logits) in enumerate(zip(model_output["teacher_logits"],
                                                model_output["student_logits"], strict=True)):
        labels = data["responses"][i]
        mask = data["response_mask"][i].to(t_logits.dtype) * data["dcrl_active"][i]
        old, advantages = data["old_log_probs"][i], data["advantages"][i]
        if not (len(labels) == len(t_logits) == len(s_logits) == len(old) == len(mask)):
            raise ValueError("DCRL response/logit alignment mismatch")
        denom = mask.sum().clamp(min=1)
        for start in range(0, len(labels), chunk_size):
            end = start + chunk_size
            token_loss, stats = checkpoint(
                dcrl_token_terms, t_logits[start:end], s_logits[start:end], labels[start:end],
                old[start:end], advantages[start:end], temperature, float(dcrl_config.clip_ratio),
                alpha, beta, use_reentrant=False,
            )
            weight = mask[start:end] / denom * scale
            loss = loss + (token_loss * weight).sum()
            diagnostics = diagnostics + (stats * weight[:, None]).sum(0)
    metrics = {f"dcrl/{name}": Metric(value=value.item(), aggregation=AggregationType.SUM)
               for name, value in zip(names, diagnostics, strict=True)}
    metrics["dcrl/joint_loss"] = Metric(value=loss.detach().item(), aggregation=AggregationType.SUM)
    return loss, metrics


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # Normalize the value loss over the global mini-batch (dp_size / batch_num_tokens /
    # global_batch_size) instead of the local micro-batch, so the accumulated critic gradient is
    # invariant to how the mini-batch is split into micro-batches (as the actor's ppo_loss does).
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]

    # When the loss is normalized over the global batch, each micro-batch contributes a partial sum,
    # so the loss metric must be aggregated with SUM to reflect the global-batch mean.
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or global_batch_size is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=config.loss_scale_factor,
    )

    metrics = {
        "critic/vf_loss": Metric(value=vf_loss, aggregation=metric_aggregation),
        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
    }

    return vf_loss, metrics
