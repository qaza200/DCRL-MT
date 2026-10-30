# DCRL-MT
Dual-Condition Reinforcement Learning for Retrieval-Augmented Low-resource Machine Translation


DCRL jointly optimizes a shared translation policy under two input conditions:

- **Base view:** translation without retrieved examples.
- **Retrieval view:** translation augmented with retrieved bilingual examples.

The model generates candidate translations under the retrieval view and evaluates the same response tokens under both views. Training combines two clipped policy objectives with cross-view KL regularization.

## Repository Structure

```text
.
├── dcrl/
│   ├── reward/
│   │   ├── formula.py
│   │   ├── metricx_server.py
│   │   ├── reward_server.py
│   │   ├── verl_reward.py
│   │   ├── run_metricx_server.sh
│   │   └── run_reward_server.sh
│   └── training/
│       ├── config.yaml
│       └── run_grpo.sh
└── verl/
    └── ...
```

The bundled `verl/` directory contains the training framework and DCRL-specific modifications. Keep `dcrl/` and `verl/` at the same directory level.

Model weights, datasets, and the external MetricX implementation are not included.

## Method Overview

For each source sentence, DCRL constructs a retrieval-augmented prompt and a base prompt.

1. Sample a group of candidate translations using the retrieval-augmented prompt.
2. Score the candidates using reference-free quality estimation and validity checks.
3. Compute group-relative advantages.
4. Evaluate the same candidate tokens under both input conditions.
5. Update the shared policy using the weighted joint objective.

The training objective is:

\[
\mathcal{L}_{\mathrm{DCRL}}
=
\alpha \mathcal{L}_{R}
+
(1-\alpha)\mathcal{L}_{B},
\]

where each view-specific loss includes a clipped policy loss and a cross-view KL term.

Both policy ratios use the old retrieval-view policy probabilities as their denominator. The KL terms are computed over the vocabulary at retrieval-sampled prefixes, with different stop-gradient directions for the two branches.

### Training Reward

The current implementation combines COMETKiwi and normalized reference-free MetricX:

\[
Q = \mathrm{COMETKiwi} + \left(1-\frac{\mathrm{MetricX}}{25}\right).
\]

Valid candidates receive \(1+Q\); invalid candidates receive \(-1\). Validity checks include empty output, response truncation, source copying, incorrect output language, and nonpositive COMETKiwi scores.

The executable reward definition is in `dcrl/reward/formula.py`. The current implementation uses the sum above; the retained `kiwi_weight` configuration field does not alter this calculation.

Groups containing no valid candidates are excluded from both policy and KL losses.

## Environment Setup

The implementation uses three environments or independently configured processes:

| Component | Main dependencies |
|---|---|
| Training | Bundled verl and its dependencies, aiohttp |
| COMETKiwi and language identification | unbabel-comet, fasttext, PyYAML |
| MetricX | MetricX-24 implementation, transformers, sentencepiece, PyYAML |

In the training environment, install the bundled framework from the repository root:

```bash
pip install -e ./verl
```

Refer to the bundled verl documentation for compatible PyTorch, CUDA, vLLM, and attention-backend installation instructions. Installing the editable package alone does not configure all accelerator dependencies.

The DCRL training path supports text-only, single-turn, synchronous GRPO with FSDP/FSDP2, sequence parallelism set to one, and unfused logits.

## Model and Data Paths

All configured resource paths are relative to the repository root. The default layout is:

```text
models/
├── Qwen3-4B-Instruct-2507/
├── wmt23-cometkiwi-da-xl/
│   └── checkpoints/model.ckpt
├── glotLID/
│   └── model_v3.bin
└── metricx-24-hybrid-xl-v2p6-bfloat16/
    └── mt5-xl-tokenizer/

third_party/
└── metricx/

data/
├── train/
│   └── train_ntrex_dcrl_k5_np.parquet
└── val/
    └── bouquet_dev.parquet
```

Download the required resources separately and update paths in:

```text
dcrl/training/config.yaml
```

The translation model and dataset paths can also be overridden through environment variables when launching training.

## Data Format

Training data must be provided as parquet files. Each row contains the retrieval-view prompt and the corresponding base-view prompt for the same source sentence.

```python
{
    "prompt": [
        {
            "role": "user",
            "content": "Retrieval-augmented prompt with bilingual examples and the source sentence."
        }
    ],
    "data_source": "translation_train",
    "reward_model": {
        "style": "custom",
        "ground_truth": ""
    },
    "extra_info": {
        "split": "train",
        "source_text": "Source sentence",
        "src_lang": "English",
        "tgt_lang": "Bengali",
        "student_prompt": [
            {
                "role": "user",
                "content": "Base translation prompt for the same source sentence."
            }
        ]
    }
}
```

Requirements:

- Store prompts as chat-message lists, not pre-rendered chat-template strings.
- The base prompt must be nonempty and no longer than the retrieval prompt after tokenization.
- Retrieved examples must already be included in the training prompt.
- Do not include the target answer for the query sentence in either prompt.
- Construct retrieval resources without evaluation-set leakage.

Retrieval and parquet construction are preprocessing steps and are not implemented by the training launcher.

Validation rows require `source_text`, `tgt_lang`, and a non-training `split` value such as `"dev"` in `extra_info`. Their main `prompt` determines the evaluated input view; `student_prompt` is not required.

Supported language-identification labels are listed in `dcrl/reward/reward_server.py`.

## Running DCRL

Run the following commands from the repository root, in separate terminals and their respective environments.

### 1. Start the MetricX Service

```bash
CUDA_VISIBLE_DEVICES=1 \
bash dcrl/reward/run_metricx_server.sh
```

The default endpoint is:

```text
http://localhost:8002
```

### 2. Start the COMETKiwi and Language-Identification Service

```bash
CUDA_VISIBLE_DEVICES=0 \
bash dcrl/reward/run_reward_server.sh
```

This service requests MetricX scores from the endpoint configured in `reward_service.metricx_url`.

The default endpoint is:

```text
http://localhost:8001
```

Check service availability:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
```

The services do not implement authentication. Keep them on localhost or a trusted private network; do not expose them to the public Internet.

### 3. Launch Training

The following example uses GPU 2 for training, separate from the two scoring GPUs. Adjust GPU assignments to your hardware.

```bash
CUDA_VISIBLE_DEVICES=2 \
MODEL_PATH=./models/Qwen3-4B-Instruct-2507 \
TRAIN_FILE=./data/train/train_ntrex_dcrl_k5_np.parquet \
VAL_FILE=./data/val/bouquet_dev.parquet \
EXPERIMENT_NAME=qwen3_4b_dcrl_joint \
bash dcrl/training/run_grpo.sh \
    trainer.dcrl.alpha=0.5
```

The checked-in configuration retains `alpha=0.0` from an ablation setting. Explicitly set `trainer.dcrl.alpha=0.5` for equal-weight joint optimization.

Selected settings can be overridden as follows:

```bash
LEARNING_RATE=5e-6 \
TRAIN_BATCH_SIZE=32 \
PPO_MINI_BATCH_SIZE=32 \
ROLLOUT_N=8 \
NGPUS_PER_NODE=1 \
bash dcrl/training/run_grpo.sh \
    trainer.dcrl.alpha=0.5 \
    trainer.dcrl.kl_coef=0.001
```

### Objective Ablations

| Setting | Configuration |
|---|---|
| Base-branch objective | `trainer.dcrl.alpha=0.0` |
| Retrieval-branch objective | `trainer.dcrl.alpha=1.0` |
| Equal-weight joint objective | `trainer.dcrl.alpha=0.5` |

All settings retain retrieval-view candidate generation. Setting `alpha=0.0` removes the retrieval-branch loss; it does not remove retrieval from the overall training procedure.

### Short Smoke Test

Before a full experiment, run a short test using a separate output directory:

```bash
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
ROLLOUT_N=4 \
VAL_BEFORE_TRAIN=False \
EXPERIMENT_NAME=dcrl_smoke \
bash dcrl/training/run_grpo.sh \
    trainer.dcrl.alpha=0.5 \
    trainer.total_training_steps=2 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    actor_rollout_ref.actor.use_dynamic_bsz=False
```

Ensure that model and data paths are configured, both scoring services are running, and GPU assignments are appropriate before executing this command.

## Outputs

By default, training artifacts are written to:

```text
outputs/<EXPERIMENT_NAME>/
├── logs/
├── train_rollouts/
├── validation_rollouts/
└── checkpoints/
```

DCRL diagnostics use these prefixes:

```text
dcrl/
actor/dcrl/
val/dcrl_cometkiwi/
```

Logged diagnostics include policy losses, KL terms, probability ratios, clipping rates, candidate validity, and quality-estimation scores.

Validation uses raw COMETKiwi as its primary score. Empty outputs have placeholder metric values; use `metricx_available` when analyzing MetricX rather than treating a placeholder error of zero as a perfect translation.

## Implementation Notes

- Retrieval and base views share policy parameters; there are no separate teacher and student optimizers.
- Response token IDs are reused without re-tokenizing the generated translations.
- Entirely inactive global mini-batches skip optimizer updates.
- The vocabulary KL at sampled prefixes is a conditional surrogate, not an exact sequence-level KL.
- Top-p sampling and repetition penalties differ from the temperature-softmax distribution used for recomputed probabilities. The importance ratios are not claimed to yield a strictly unbiased estimator.
- Renamed configuration keys and diagnostic fields use the `dcrl` namespace.
- Static checks do not replace GPU and distributed-training validation.

## Acknowledgements

This implementation builds on [verl](https://github.com/verl-project/verl) and uses [COMET](https://github.com/Unbabel/COMET), [MetricX](https://github.com/google-research/metricx), and [GlotLID](https://github.com/cisnlp/GlotLID).

Please retain the upstream license and copyright notices. Third-party code, models, and datasets remain subject to their respective licenses.
