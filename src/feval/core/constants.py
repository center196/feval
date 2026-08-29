from __future__ import annotations


PROTOCOL_NETWORK = "feval-network-v28"
PROTOCOL_MODEL_COMMITMENT = "f3"
PROTOCOL_MODEL_MANIFEST = "feval-model-v4"
PROTOCOL_ROLLOUT_MANIFEST = "feval-rollouts-v17"
PROTOCOL_VALIDATOR_STATE = "feval-validator-state-v29"
PROTOCOL_MINER_ROLLOUT_STATE = "feval-miner-rollout-state-v14"

SUBNET_NETUID = 47
PUBLIC_WANDB_ENTITY = "feval196-feval"
PUBLIC_WANDB_PROJECT = "feval-valid"
CHAMPION_COUNT = 1
PROMOTION_DELTA_MIN = 0.01
PROMOTION_CONFIDENCE_Z = 2.326347874
CHAMPION_SHARES = (0.10,)
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

MATH_DATASET = "nvidia/Nemotron-SFT-Math-v4"
INSTRUCTION_DATASET = "nvidia/Nemotron-RL-instruction_following"
# Immutable source snapshots. Changing either is a subnet protocol upgrade.
MATH_DATASET_REVISION = "84d42ad0cb960f07f951b9baa9ed2b46a5a18c66"
INSTRUCTION_DATASET_REVISION = "3b253899665cb71334bb54c14eb5d91751beaad7"

EVALUATION_ROWS = 10_000
MATH_ROWS = 5_000
INSTRUCTION_ROWS = 5_000
CANDIDATE_POOL_ROWS_PER_TASK = 50_000
DATASET_WINDOW_BLOCKS = 3_600
WEIGHT_INTERVAL_BLOCKS = 150
AUDIT_DELAY_BLOCKS = 5
AUDIT_ROWS_PER_ROUND = 32
# A miner submits 10K rows. Thirty-two distinct uniformly random rows are
# checked per round. Ten successful rounds check 320 rows and provide at least
# a 95% conservative guarantee (96.19% exact without-replacement detection)
# when 1% or more are forged.
# Align the audit threat model with the smallest score movement that can clear
# the king-promotion margin. Fabricating fewer rows cannot create a full margin
# from an otherwise tied model.
AUDIT_MIN_FAKE_ROW_FRACTION = PROMOTION_DELTA_MIN
AUDIT_DETECTION_CONFIDENCE = 0.95
# Ten successful rounds make a revision eligible for emission. Continue
# auditing that exact immutable revision until twenty rounds have passed so
# eligibility is not also the end of ongoing fraud detection.
AUDIT_TOTAL_ROUNDS = 20
# Miners decode greedily, so every audited token must be the exact rank-one
# token under the committed adapter. A merely plausible near-top token does
# not prove that the submitted trace is the model's greedy rollout.
AUDIT_MIN_RELATIVE_PROBABILITY = 1.0
AUDIT_MIN_EXACT_ARGMAX_RATIO = 1.0
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
