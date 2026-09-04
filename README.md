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

Each two-day window draws its 10,000 rows from five pinned public Hugging Face
datasets. Sources are never downloaded whole: parquet footers give a row-group
map for free, and the window seed picks which row groups (or, for a JSONL
source, which byte blocks) to read, fetching only the columns Feval grades on.
Building a window transfers a few hundred megabytes rather than tens of
gigabytes, and takes roughly a minute and a half. The first cycle for a window
reports per-source scan progress; the unauthenticated-request warning is
harmless for public repositories.

## Repository layout

```text
.
+-- src/feval/
|   +-- cli/          # Command-line interface
|   +-- core/         # Network config and protocol constants
|   +-- protocol/     # Deterministic seeds, Merkle roots, and submissions
|   +-- datasets/     # Source sampling, task normalisation, and verifiers
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

The published leaderboard table includes each miner's `model_repo`,
`rollout_repo`, and `invalid_reason`. The invalid reason is empty while a miner
is still auditing or an operational retry is in progress. It contains only a
sanitized audit-gate outcome; internal system errors are not published.

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
derive the same 10,000-row evaluation set from immutable, code-pinned dataset
revisions. Every row is settled by exact string comparison. There is no model
judge, no symbolic algebra, and no execution of dataset or model code anywhere
in the grading path.

| Source | Rows | Verifier |
| --- | --- | --- |
| nvidia/OpenMathReasoning | 3,000 | `strict_numeric` |
| nvidia/Nemotron-CrossThink | 1,500 | `strict_numeric` |
| nvidia/Nemotron-RL-knowledge-mcqa | 2,000 | `mcqa_letter` |
| nvidia/OpenScienceReasoning-2 | 1,500 | `mcqa_letter` |
| PrimeIntellect/synthetic-code-understanding | 2,000 | `json_output_exact` |

The three verifiers are:

- `strict_numeric` compares complete integers, decimals, and fractions by exact
  `Fraction` equality. Expressions, variables, and prose are rejected rather
  than approximated.
- `mcqa_letter` compares one option letter. Only ten-option questions are kept,
  and Knowledge-MCQA's declared option keys must agree with the options rendered
  in the prompt. Static local parsing accepts the source layouts `A:`, `A.`, and
  `A)`; source-provided regular expressions are never compiled. Blind guessing
  is therefore worth 10% on these rows rather than 25%.
- `json_output_exact` requires one complete JSON object with exactly one string
  field named `output`, then compares that field byte for byte, including its
  leading and trailing whitespace. The program is never run; the expected
  output ships with the pinned revision.

6,500 of the 10,000 rows use a verifier where guessing is worthless, and the
protocol refuses to build a window that falls below that floor. A miner that
answers every row with the single most common option letter scores about 6%.

Each source keeps its own answer conventions out of the prompt: Feval extracts
the bare question, supplies its own output instruction, and drops any row whose
text still carries a competing format instruction. CrossThink rows that visibly
request multiple answers are also dropped because one scalar ground-truth field
cannot grade them unambiguously. Dataset text and metadata are handled only as
bounded data: they are never evaluated, compiled, imported, or executed.

Production commands cannot replace those datasets or reduce the row count.
Miners select `--max-new-tokens` from 1 through 32,768; validators read the
rollout manifest and enforce that exact requested limit. Prompt plus response
is always capped at 32,768 tokens, so the effective generation budget for each
row is the smaller of the miner's choice and the context remaining after its
fixed prompt. A 256 MiB total bundle cap prevents miners from imposing
multi-gigabyte downloads. Dataset, verifier, and context changes are manual
protocol upgrades; `sources_digest` in the network config pins every repository,
revision, file, column, verifier, and row quota as one value. Validators score
all rows locally, then verify unpredictable samples drawn only from correctly
answered rows against the committed adapter using bounded
greedy-token checks. Every token must be within the top three and a 0.25
logprob gap, while at least 99.5% must be exact rank one. Eligibility normally
requires 10 successful rounds of 32 distinct correct rows; smaller correct
populations are fully covered sooner, and auditing continues for up to 20 rounds.
Validators consider only currently registered miners with valid Feval
metadata. Exact model and exact full-rollout copies belong to the earlier
current commitment block, with the hotkey as a deterministic same-block tie
breaker. Replacing a commitment or leaving the current miner set removes its
priority.

The math checker is deliberately narrower than a full symbolic grader: it
accepts only complete numeric answers, which drops most of the symbolic rows in
the source datasets. That is the intended trade. It keeps validation
deterministic across GPUs and runtimes, inexpensive, and free of dataset code
execution, and it means two validators can never disagree about a score. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dataset licences and
attribution obligations.

## Decentralization

Feval has two operating roles only:

- Miners commit models on chain and publish immutable model and rollout data.
- Validators independently read finalized chain state, derive each evaluation
  set from seeded row groups of the pinned public dataset revisions, audit
  miners by sampling only score-contributing correct rows, and submit weights.

There is no third operational service, privileged protocol key, miner allowlist,
or centrally supplied launch configuration. The finalized chain, immutable source
revisions, and versioned protocol code are the shared inputs. Every evaluation
manifest includes a deterministic root that validators can compare directly.

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
