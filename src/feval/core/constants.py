from __future__ import annotations


PROTOCOL_NETWORK = "feval-network-v37"
PROTOCOL_MODEL_COMMITMENT = "f3"
PROTOCOL_MODEL_MANIFEST = "feval-model-v4"
PROTOCOL_ROLLOUT_MANIFEST = "feval-rollouts-v20"
PROTOCOL_VALIDATOR_STATE = "feval-validator-state-v35"
PROTOCOL_MINER_ROLLOUT_STATE = "feval-miner-rollout-state-v18"

SUBNET_NETUID = 47
PUBLIC_WANDB_ENTITY = "feval196-feval"
PUBLIC_WANDB_PROJECT = "feval-valid"
CHAMPION_COUNT = 1
PROMOTION_DELTA_MIN = 0.01
PROMOTION_CONFIDENCE_Z = 2.326347874
MINER_SHARE = 0.10
BURN_SHARE = 0.90
CHALLENGER_SHARE = 0.0
INVALID_ROUNDS_BEFORE_BLACKLIST = 3
BLACKLIST_ENABLED = False
# Bittensor targets roughly one finalized block per 12 seconds. Protocol time
# penalties use block height so every validator reaches the same expiry point.
BLACKLIST_DURATION_BLOCKS = 7 * 24 * 60 * 5

BASE_MODEL = "Qwen/Qwen3-4B-Base"
# Pin the base model and tokenizer. Changing this is a subnet protocol upgrade.
BASE_MODEL_REVISION = "b1ecb62b18378adb8af264260fbbbd618222be87"
# Architecture dimensions from the pinned Qwen3-4B-Base config. They are
# repeated here so malformed miner tensor shapes are rejected on CPU before a
# native GPU loader sees them.
BASE_HIDDEN_SIZE = 2_560
BASE_ATTENTION_SIZE = 4_096
BASE_KEY_VALUE_SIZE = 1_024
BASE_INTERMEDIATE_SIZE = 9_728
BASE_NUM_HIDDEN_LAYERS = 36

# Immutable evaluation sources. Every entry is pinned to a full commit SHA and
# to the exact files and columns Feval is allowed to read, so each node fetches
# byte-identical input. Changing any field is a subnet protocol upgrade.
#
# Only tasks a validator can settle with pure string work are listed. There is
# no model judge, no symbolic algebra, and no execution of dataset or model
# code anywhere in the grading path.
EVALUATION_SOURCES: tuple[dict, ...] = (
    {
        "name": "open_math_reasoning",
        "kind": "parquet",
        "repo": "nvidia/OpenMathReasoning",
        "revision": "d3d08664755704f422af97d43a7ff0ded4bd95df",
        "files": tuple(f"data/cot-{index:05d}-of-00144.parquet" for index in range(144)),
        "columns": ("problem", "expected_answer", "problem_type"),
        "verifier": "strict_numeric",
        "license": "cc-by-4.0",
        "rows": 3_000,
    },
    {
        "name": "crossthink_math",
        "kind": "jsonl",
        "repo": "nvidia/Nemotron-CrossThink",
        "revision": "a4ce9a3b9434c5f231e2cbe30696d9a721c11d69",
        "files": ("Data/Nemotron-CrossThink-Math.jsonl",),
        "columns": (),
        "verifier": "strict_numeric",
        "license": "cc-by-4.0",
        "rows": 1_500,
    },
    {
        "name": "knowledge_mcqa",
        "kind": "parquet",
        "repo": "nvidia/Nemotron-RL-knowledge-mcqa",
        "revision": "62a1eec1f952723eab2ee3832222f533b8138067",
        "files": tuple(f"data/train-{index:05d}-of-00004.parquet" for index in range(4)),
        "columns": (
            "responses_create_params",
            "expected_answer",
            "uuid",
            "template_metadata",
            "options",
        ),
        "verifier": "mcqa_letter",
        "license": "cc-by-4.0",
        "rows": 2_000,
    },
    {
        "name": "open_science",
        "kind": "parquet",
        "repo": "nvidia/OpenScienceReasoning-2",
        "revision": "174b02c9cdf231f220765b2a1d5ece4550921894",
        "files": ("train/OpenScienceReasoning-2.parquet",),
        # 'output' holds a long reasoning trace Feval never reads. Omitting it
        # takes a row group from 1.9 GB to about 133 MB on the wire.
        "columns": ("expected_answer", "input"),
        "verifier": "mcqa_letter",
        "license": "cc-by-4.0",
        "rows": 1_500,
    },
    {
        "name": "code_understanding",
        "kind": "parquet",
        "repo": "PrimeIntellect/synthetic-code-understanding",
        "revision": "106a1cec075ae29b8dc07e355a29ddce2cf0745b",
        "files": ("data/train-00000-of-00001.parquet",),
        "columns": ("problem_id", "prompt", "verification_info"),
        "verifier": "json_output_exact",
        # Apache-2.0 through PrimeIntellect/SYNTHETIC-1, which names this repo
        # in its own hf_dataset_name column. See THIRD_PARTY_NOTICES.md.
        "license": "apache-2.0",
        "rows": 2_000,
    },
)

EVALUATION_ROWS = 10_000
# Rows whose answer space makes blind guessing worthless. Keeping this the
# clear majority bounds the score a miner can reach without a real model.
GUESS_RESISTANT_ROWS = 6_500
# Keep each deterministic 10,000-row evaluation set active for approximately
# two days at Bittensor's target of one finalized block every 12 seconds.
DATASET_WINDOW_BLOCKS = 2 * 24 * 60 * 5
WEIGHT_INTERVAL_BLOCKS = 150
AUDIT_DELAY_BLOCKS = 5
AUDIT_ROWS_PER_ROUND = 32
# Only correctly answered rows can contribute to a miner's score, so audit
# sampling is restricted to that population. Thirty-two distinct correct rows
# are checked per round. Ten successful rounds check 320 rows and provide at
# least a 95% conservative guarantee when 1% or more of the score-contributing
# rows are forged. Incorrect rows cannot improve a miner's score.
AUDIT_MIN_FAKE_ROW_FRACTION = PROMOTION_DELTA_MIN
AUDIT_DETECTION_CONFIDENCE = 0.95
# Ten successful rounds make a revision eligible for emission. Continue
# auditing that exact immutable revision until twenty rounds have passed so
# eligibility is not also the end of ongoing fraud detection.
AUDIT_TOTAL_ROUNDS = 20
# BF16 decode and teacher-forced kernels can reorder a few close logits across
# supported GPUs. Bound that numerical tolerance while requiring almost every
# audited token to remain the exact rank-one choice.
AUDIT_MAX_TOKEN_RANK = 3
AUDIT_MIN_RELATIVE_PROBABILITY = 0.7788007830714049  # exp(-0.25)
AUDIT_MIN_EXACT_ARGMAX_RATIO = 0.995
# Validator audit memory profile. Keep one active audit sequence and chunk
# prompt-logprob work so several maximum-length traces cannot accumulate their
# KV caches concurrently.
AUDIT_GPU_MEMORY_UTILIZATION = 0.60
AUDIT_MAX_NUM_BATCHED_TOKENS = 4_096
AUDIT_MAX_NUM_SEQS = 1

MAX_PROMPT_CHARS = 32_768
MAX_OUTPUT_TOKENS = 32_768
# This is a hard combined prompt-plus-response ceiling. A miner may choose any
# lower generation budget, but each row receives only the context remaining
# after the protocol-owned prompt is tokenized.
MAX_CONTEXT_TOKENS = 32_768
DEFAULT_ROLLOUT_BATCH_SIZE = 4
MAX_LORA_RANK = 16
MAX_LORA_ALPHA = 256
MAX_ABS_LORA_VALUE = 100.0
MAX_ADAPTER_BYTES = 256 * 1024 * 1024
MAX_ADAPTER_ELEMENTS = 100_000_000
MAX_TENSOR_DIMENSION = 65_536
MAX_ADAPTER_CONFIG_BYTES = 64 * 1024
# The aggregate cap remains deliberately much smaller than 10,000 worst-case
# 32K rows. This bounds untrusted downloads and host-memory use while allowing
# individual long reasoning traces; miners must terminate ordinary rows early.
MAX_ROLLOUT_BYTES = 256 * 1024 * 1024
MAX_ROLLOUT_LINE_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 64 * 1024

# Only these miner-controlled files are ever downloaded. Python, pickle, model
# configuration, tokenizer, templates, and dataset scripts are never loaded
# from miner repositories.
MODEL_FILES = ("adapter_config.json", "adapter_model.safetensors")
ROLLOUT_FILES = ("manifest.json", "rollouts.jsonl")

ALLOWED_LORA_TARGET_MODULES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


def _sources_digest() -> str:
    """Pin the whole source table so a local config cannot rewrite one field."""

    import hashlib

    from ..utils.jsonutil import canonical_json_bytes

    return hashlib.sha256(
        canonical_json_bytes(
            [
                {key: list(value) if isinstance(value, tuple) else value
                 for key, value in sorted(spec.items())}
                for spec in EVALUATION_SOURCES
            ]
        )
    ).hexdigest()


EVALUATION_SOURCES_DIGEST = _sources_digest()
