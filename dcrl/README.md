# DCRL

Dual-condition reinforcement learning for machine translation. A shared policy
generates candidates under retrieval-augmented prompts and scores the same response
tokens under base prompts. The implementation combines two clipped policy losses
and conditional full-vocabulary KL terms with different stop-gradient directions.

## Repository layout

Place these two directories directly at the repository root:

```text
dcrl/
  reward/
  training/
verl/
```

No additional outer directory name is required. The bundled `verl/` already
contains the DCRL implementation; do not apply an additional patch.
Retain the upstream licenses and copyright notices in `verl/`.

## Paths and environments

Run commands from the repository root. Model and dataset paths in
`dcrl/training/config.yaml` and the launcher are relative to that root.
Place model weights in `./models/`, datasets in `./data/`, and the MetricX source
checkout in `./third_party/metricx`, or change the relative paths accordingly.
Weights and datasets are not included in this code release.

Use separate environments for the trainer, COMETKiwi/GlotLID service, and MetricX
service. The trainer requires the bundled verl dependencies and aiohttp;
the COMET service requires unbabel-comet, fasttext and PyYAML; the MetricX
service requires metricx24, transformers, sentencepiece and PyYAML.
Install the bundled trainer with `pip install -e ./verl` in its environment.

The launchers locate the repository root from their own location and do not
search for an external sibling verl checkout. Runtime libraries may receive
absolute paths computed from the current checkout; no machine-specific absolute
path is embedded in the configuration.

## Dataset schema

Training parquet rows must contain a retrieval prompt and the corresponding
base prompt for the same source sentence:

```python
{
    "prompt": [{"role": "user", "content": "Retrieval-augmented translation prompt"}],
    "data_source": "translation_train",
    "reward_model": {"style": "custom", "ground_truth": ""},
    "extra_info": {
        "split": "train",
        "source_text": "Source sentence",
        "src_lang": "English",
        "tgt_lang": "Bengali",
        "student_prompt": [{"role": "user", "content": "Base translation prompt"}]
    }
}
```

Prompts are chat-message lists, not already-rendered chat-template strings.
The base prompt must be nonempty and no longer than the retrieval prompt.
Validation rows use the desired inference prompt and require `source_text`,
`tgt_lang` and a non-training `split` in `extra_info`.

## Launch

In the MetricX environment:

```bash
CUDA_VISIBLE_DEVICES=1 bash dcrl/reward/run_metricx_server.sh
```

In the COMET environment:

```bash
CUDA_VISIBLE_DEVICES=0 bash dcrl/reward/run_reward_server.sh
```

In the trainer environment, select training GPUs separately from scoring GPUs:

```bash
MODEL_PATH=./models/Qwen3-4B-Instruct-2507 \
TRAIN_FILE=./data/train/train_ntrex_dcrl_k5_np.parquet \
VAL_FILE=./data/val/bouquet_dev.parquet \
bash dcrl/training/run_grpo.sh
```

Both scoring services default to localhost. For multi-machine operation,
configure the service endpoints through your private deployment configuration
and environment variables; do not publish private hostnames or addresses.
The services have no authentication and must not be exposed to the public Internet.

Method configuration uses `trainer.dcrl.*`. The existing alpha and reward-weight
settings are preserved; select the intended experimental setting explicitly.
This release cleanup does not change the reward calculation in `reward/formula.py`.
Service configuration changes require restarting the corresponding service.

## Implementation boundaries

- Text-only, single-turn, synchronous GRPO training with FSDP/FSDP2; SP=1 and
  unfused logits are required.
- Retrieval-view candidates are reused under both prompts. Both policy ratios
  use recomputed old retrieval-view log-probabilities as the denominator.
- KL is computed over the vocabulary at sampled prefixes, not as an exact
  sequence-distribution KL. Losses use sequence-mean/token-mean aggregation.
- Entirely invalid candidate groups do not contribute policy or KL losses;
  entirely inactive global mini-batches skip optimizer updates.
- Top-p sampling and repetition penalties differ from the temperature-softmax
  distribution used for recomputed probabilities; the estimator is not claimed
  to be strictly unbiased.
- Validation reports raw COMETKiwi. Empty-output MetricX placeholders must be
  excluded using `metricx_available` when analyzing MetricX.
- LID support is explicitly enumerated in `reward/reward_server.py`.

Logs use the `dcrl/`, `actor/dcrl/`, and `val/dcrl_cometkiwi/` prefixes.
Old configuration keys and log-field names are not aliases in this release.

Before a full run, validate dependencies and data, check service health, and
perform a short training smoke test with a fresh output directory. Static checks
do not establish GPU or distributed-training correctness.
