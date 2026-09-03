# Feval

Feval is a Bittensor subnet for evaluating LoRA improvements to a fixed base
model. Miners publish a model and its rollouts, validators verify the work and
independently set weights from the same code-pinned protocol rules.

## Install

Use Linux or WSL with an NVIDIA GPU, a working CUDA driver, Python 3.12, and a
Bittensor wallet.

```bash
git clone <repository-url> feval
cd feval
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
feval --help
```

Copy the environment template only when credentials or reporting settings are
needed:

```bash
cp .env.example .env
```

The CLI reads `.env` from the current working directory before running a
command. Run it from the directory containing your `.env`; values already
exported in the shell take precedence over that file.

Set `HF_TOKEN` on miner hosts that upload to Hugging Face. Validators do not
need a Hugging Face token when model and rollout repositories are public.
Never commit `.env`, wallet files, private keys, tokens, or validator state.

## Repository layout

```text
.
+-- src/feval/
|   +-- cli/          # Command-line interface
|   +-- core/         # Network config and protocol constants
|   +-- protocol/     # Deterministic seeds, Merkle roots, and submissions
|   +-- datasets/     # Dataset loading, filtering, and scoring
|   +-- models/       # LoRA artifacts, rollout generation, and inference
|   +-- chain/        # Bittensor commitments, axon serving, and weights
|   +-- nodes/        # Miner and validator runtime loops
|   `-- utils/        # Small shared helpers
+-- docs/assets/      # Documentation assets
+-- network.json      # Reviewed local overrides for code-pinned defaults
+-- SECURITY.md       # Trust model and validator hardening notes
+-- THIRD_PARTY_NOTICES.md # Dataset attribution and license summary
`-- pyproject.toml    # Package metadata and dependencies
```

## Miner

A miner trains a PEFT LoRA for the base model pinned in `network.json`. The
adapter directory must contain:

```text
adapter_config.json
adapter_model.safetensors
```

Create one Hugging Face model repository and one Hugging Face dataset
repository for rollouts. Publish the adapter and commit its immutable revision
on chain:

```bash
feval miner publish-model \
  --config network.json \
  --adapter-dir output/my-lora \
  --model-repo <hf-account>/<model-repo> \
  --rollout-repo <hf-account>/<rollout-repo> \
  -w <wallet-name> -H <hotkey-name> -n finney
```

Keep the current evaluation-window rollouts updated:

```bash
feval miner watch-rollouts \
  --config network.json \
  --adapter-dir output/my-lora \
  --work-dir miner-work \
  --max-new-tokens 32768 \
  --batch-size 4 \
  --poll-seconds 60 \
  -w <wallet-name> -H <hotkey-name> -n finney
```

The watcher uploads a new rollout when the evaluation window or committed
model changes. Publish the model again before generating rollouts whenever the
adapter bytes change. Use `feval miner status --help` and
`feval miner leaderboard --help` to inspect public validator results.

## Validator

A validator reads miner commitments, pins immutable Hugging Face revisions,
scores each complete rollout set, audits sampled token traces with the
committed LoRA, and submits weights.

The `feval196-feval/feval-valid` W&B project is Open, so validators do not need
an invitation or team membership. Each validator authenticates with its own
W&B account by setting `WANDB_API_KEY`, and Feval automatically publishes its
public result summary to this fixed destination. Reporting failures are visible
in validator status but never change evaluation or emitted weights.

Set `FEVAL_REPORT_WANDB=0` in `.env` to disable automatic reporting. The public
destination is `WANDB_ENTITY=feval196-feval` and `WANDB_PROJECT=feval-valid`;
validator results always use that shared destination.

Run continuously:

```bash
feval validator run \
  --config network.json \
  --work-dir validator-work \
  --state validator-state.json \
  -w <wallet-name> -H <hotkey-name> -n finney
```

Feval automatically starts vLLM workers with `spawn` and uses the native sampler
to avoid fork-related startup hangs and FlashInfer sampler warmup failures.
No environment exports are needed for these defaults. Explicit
`VLLM_WORKER_MULTIPROC_METHOD` and `VLLM_USE_FLASHINFER_SAMPLER` settings still
override them.

Keep `validator-state.json` and its backup between restarts. To verify a setup
without submitting weights, run one cycle with:

```bash
feval validator run \
  --config network.json \
  --work-dir validator-work \
  --state validator-state.json \
  --once --dry-run-weights \
  -w <wallet-name> -H <hotkey-name> -n <network>
```

Check liveness with `feval health --state validator-state.json`. See
`feval validator export-results --help` to publish a sanitized result summary.

## Evaluation protocol

Every 14,400 finalized blocks (approximately two days), miners and validators
derive the same 10,000-row evaluation set from immutable, code-pinned NVIDIA
dataset revisions:

- 5,000 math rows whose expected answers can be safely verified as complete
  integers, decimals, or fractions.
- 5,000 instruction-following rows using binary all-constraints-pass grading
  from a reviewed set of 30 local deterministic predicates.

At the pinned instruction revision, 16,377 of 46,391 source rows pass the safe
schema, constraint, and prompt filters, leaving substantial headroom above the
5,000-row window requirement.

Production commands cannot replace those datasets or reduce the row count.
Miners select `--max-new-tokens` from 1 through 32,768; validators read the
rollout manifest and enforce that exact requested limit. Prompt plus response
is always capped at 32,768 tokens, so the effective generation budget for each
row is the smaller of the miner's choice and the context remaining after its
fixed prompt. A 256 MiB total bundle cap prevents miners from imposing
multi-gigabyte downloads. Dataset and context changes are manual protocol
upgrades. Validators score all rows locally, then verify unpredictable samples
against the committed adapter using bounded
greedy-token checks. Every token must be within the top three and a 0.25
logprob gap, while at least 99.5% must be exact rank one. Eligibility requires
10 successful rounds of 32 distinct rows; auditing continues for 20 rounds.
Validators consider only currently registered miners with valid Feval
metadata. Exact model and exact full-rollout copies belong to the earlier
current commitment block, with the hotkey as a deterministic same-block tie
breaker. Replacing a commitment or leaving the current miner set removes its
priority.

The local math checker is deliberately narrower than NVIDIA's full symbolic
math tooling, and the instruction checker implements only its reviewed safe
subset. This keeps validation deterministic, inexpensive, and free of dataset
code execution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dataset
licenses and attribution obligations.

## Decentralization

Feval has two operating roles only:

- Miners commit models on chain and publish immutable model and rollout data.
- Validators independently read finalized chain state, derive the evaluation
  pool from pinned public dataset revisions, audit miners, and submit weights.

There is no third operational service, privileged protocol key, miner allowlist,
or centrally supplied launch configuration. The finalized chain, immutable source
revisions, and versioned protocol code are the shared inputs. Validators can
compare their independently derived candidate-pool roots with:

```bash
feval dataset candidate-pool \
  --config network.json \
  --out validator-work/candidate-pool.jsonl \
  --manifest validator-work/candidate-pool.manifest.json
```

Reporting services such as Weights & Biases are optional mirrors and never
affect scoring or consensus. Protocol upgrades take effect only when miners and
validators choose to run the same reviewed protocol version.

`network.json` is a convenient reviewed copy of the defaults. If it is absent,
the CLI uses the same versioned defaults from the installed code. Any supplied
file is validated against consensus-critical constants before use.

## Security

Miner repositories are untrusted. Validators accept only bounded JSON,
JSONL, and SafeTensors inputs and must never run code from miner repositories.
Keep credentials in local environment variables or `.env`, use least-privilege
Hugging Face tokens, protect Bittensor wallets, and keep operational work and
state files outside version control.

See [SECURITY.md](SECURITY.md) for the full trust model and operational
hardening guidance. Use `feval miner --help`, `feval validator --help`, and
`feval dataset --help` for all command options.
