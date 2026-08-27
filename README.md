# Feval

Feval is a Bittensor subnet where miners publish LoRA models and validators
check their rollouts and assign weights.

## Install

Requires Linux or WSL, Python 3.12, an NVIDIA GPU, and a Bittensor wallet.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
feval --help
```

Miners must copy `.env.example` to `.env` and set `HF_TOKEN` for Hugging Face
uploads. Do not commit `.env`.

## Miner

A miner trains a LoRA adapter, publishes it, and keeps its rollout repository
updated. The adapter directory must contain `adapter_config.json` and
`adapter_model.safetensors`.

Publish the model:

```bash
feval miner publish-model --adapter-dir output/my-lora --model-repo my-org/my-lora --rollout-repo my-org/my-rollouts -w my-wallet -H my-hotkey -n finney
```

Keep rollouts updated:

```bash
feval miner watch-rollouts --adapter-dir output/my-lora --work-dir miner-work -w my-wallet -H my-hotkey -n finney
```

## Validator

A validator reads miner commitments, checks their models and rollouts, and
submits weights.

```bash
feval validator run --work-dir validator-work --state validator-state.json -w validator-wallet -H validator-hotkey -n finney
```

Keep `validator-state.json` between restarts.

## Help

```bash
feval miner --help
feval validator --help
```

See [SECURITY.md](SECURITY.md) for security guidance.
