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

Each twelve-hour window draws 100,000 rows from six pinned public Hugging Face
datasets whose raw candidate union contains well over one million rows. Sources
are never downloaded whole: parquet footers give a row-group map for free, and
a seed revealed at the window boundary picks which row groups (or, for a JSONL
source, which byte blocks) to read, fetching only the columns Feval grades on.
The first cycle for a window reports per-source scan progress; the
unauthenticated-request warning is harmless for public repositories.

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

Every 3,600 finalized blocks (approximately twelve hours), miners and validators
derive the same 100,000-row evaluation set from immutable, code-pinned dataset
revisions. A model must be committed before the first block of the window. The
hash of that finalized boundary block reveals the evaluation seed, so a miner
cannot know the exact next sample before committing its model.

The per-window source mixture is deterministic and has no manually maintained
category ratio. Rows are allocated by largest remainder in direct proportion
to the pinned size of each selected source split:

| Source | Pinned source rows | Rows per window |
| --- | ---: | ---: |
| OpenMathReasoning `cot` | 3,201,061 | 56,382 |
| Nemotron-CrossThink Math | 99,880 | 1,759 |
| NuminaMath-1.5 `train` | 896,215 | 15,785 |
| Nemotron-RL-knowledge-mcqa `train` | 617,020 | 10,868 |
| OpenScienceReasoning-2 | 802,666 | 14,138 |
| synthetic-code-understanding | 60,621 | 1,068 |

The protocol pins the Hub viewer's 802,666-row OpenScience estimate because its
single original Parquet file is only partially indexed by that viewer.
Obtaining post-normalization usable counts would require scanning every source,
which the runtime deliberately does not do. A window fails closed if seeded
partial reads cannot fill any derived quota. Even on failure, the reader is
limited to 90% of a source's independently addressable row groups or byte
blocks (a one-block file is the unavoidable minimum). Every selected row has
equal score weight and is settled by deterministic normalization followed by
exact equality. There is no model judge, no symbolic
algebra, and no execution of dataset or model code anywhere in the grading
path.

Rows may recur across independently sampled windows. Feval deliberately treats
the public corpus as trainable: the boundary seed hides the exact next mixture,
not the source data. A strict 180-day no-reuse rule would require 36 million
distinct eligible rows at two 100,000-row windows per day, which the currently
verified pool cannot support.

The three verifiers are:

- `math_exact` statically extracts the final answer from a complete response,
  the last balanced `\boxed{...}`,
  an `<answer>...</answer>` body, or an `Answer`/`Final answer` marker. Both
  sides receive presentation-only whitespace and outer-wrapper normalization,
  then are compared exactly. Answer-bearing symbolic expressions, variables,
  and multiple-value answers remain eligible. Nothing is simplified or
  evaluated: `0.5` differs from `1/2`, and `x+y` differs from `y+x`. Proof and
  missing-answer rows are excluded.
- `mcqa_letter` compares one option letter. Only ten-option questions are kept,
  and Knowledge-MCQA's declared option keys must agree with the options rendered
  in the prompt. Static local parsing accepts the source layouts `A:`, `A.`, and
  `A)`; source-provided regular expressions are never compiled. Blind guessing
  is therefore worth 10% on these rows rather than 25%.
- `json_output_exact` permits a long reasoning trace, statically extracts the
  final JSON object having exactly one string field named `output`, then
  compares that field byte for byte, including its leading and trailing
  whitespace. Reasoning and program text remain inert and are never run; the
  expected output ships with the pinned revision.

Each row contributes equally to the overall exact-match score. Per-source and
per-category counts and scores are retained as diagnostics, but there is no
manual category weighting or category score floor.

Each source keeps its own answer conventions out of the prompt: Feval extracts
the bare question, supplies its own output instruction, and drops any row whose
text still carries a competing format instruction. Dataset text and metadata are handled only as
bounded data: they are never evaluated, compiled, imported, or executed.

Production commands cannot replace those datasets or reduce the row count.
Miners select `--max-new-tokens` from 1 through 32,768; validators read the
rollout manifest and enforce that exact requested limit. Prompt plus response
is always capped at 32,768 tokens, so the effective generation budget for each
row is the smaller of the miner's choice and the context remaining after its
fixed prompt.

The future boundary seed independently assigns each row one of five reasoning
levels: 1,024, 2,048, 4,096, 8,192, or 16,384 tokenizer tokens. The shared
system prompt requires one response-leading `<think>...</think>` block. Only
tokens inside that first block count, and ±10% of the assigned level is valid.
Text after `</think>` has no separate length target and is the only text passed
to the answer verifier. Missing, malformed, or out-of-range reasoning blocks
score zero. Miners generate with this prompt; validators reconstruct the same
prompt, recount with the pinned tokenizer, and teacher-force the complete
rollout during audits. The rollout bundle is capped at 8 GiB because mandatory
long reasoning makes the former 256 MiB ceiling impossible.

Dataset, verifier, and context changes are manual
protocol upgrades; `sources_digest` in the network config pins every repository,
revision, file, column, verifier, category, and selected-split size as one
value. Validators calculate one exact-match average across all rows and also
publish per-category diagnostics. They then verify unpredictable samples drawn only
from correctly answered rows against the committed adapter using bounded
greedy-token checks. Audit sampling is uniform across the remaining correctly
answered rows. Every token must be within the top three and a 0.25 logprob gap, while
at least 99.5% must be exact rank one. Eligibility normally requires 30
successful rounds of 32 distinct correct rows (960 rows total), which detects a
1% forged correct-row population with greater than 99.99% probability. After
every participating miner has either reached 30 rounds or terminated, valid
miners are re-audited until 50 successful rounds. Smaller correct populations
are fully covered sooner.
Validators consider only currently registered miners with valid Feval
metadata. Exact model and exact full-rollout copies belong to the earlier
current commitment block, with the hotkey as a deterministic same-block tie
breaker. Replacing a commitment or leaving the current miner set removes its
priority.

The math checker is deliberately an exact matcher rather than a symbolic
grader. It retains any answer-bearing math row but does not determine whether
two different expressions are mathematically equivalent. This keeps validation
deterministic across GPUs and runtimes and free of dataset code execution. See
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
