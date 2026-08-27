from __future__ import annotations


PROTOCOL_NETWORK = "feval-network-v9"
PROTOCOL_MODEL_COMMITMENT = "f3"
PROTOCOL_MODEL_MANIFEST = "feval-model-v3"
PROTOCOL_ROLLOUT_MANIFEST = "feval-rollouts-v8"
PROTOCOL_VALIDATOR_STATE = "feval-validator-state-v10"
PROTOCOL_MINER_ROLLOUT_STATE = "feval-miner-rollout-state-v5"

SUBNET_NETUID = 47
CHAMPION_COUNT = 3
PROMOTION_DELTA_MIN = 0.01
PROMOTION_CONFIDENCE_Z = 2.326347874
CHAMPION_SHARES = (0.50, 0.20, 0.10)
CHALLENGER_SHARE = 0.0
HISTORY_BATCH_BLOCKS = 10_000

BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
# Pin the base model and tokenizer. Changing this is a subnet protocol upgrade.
BASE_MODEL_REVISION = "daa9c16f371249f9ad1c75a9ed6f956c08ea08f5"

MATH_DATASET = "nvidia/Nemotron-SFT-Math-v4"
INSTRUCTION_DATASET = "nvidia/Nemotron-RL-instruction_following"
# Immutable source snapshots. Changing either is a subnet protocol upgrade.
MATH_DATASET_REVISION = "84d42ad0cb960f07f951b9baa9ed2b46a5a18c66"
INSTRUCTION_DATASET_REVISION = "3b253899665cb71334bb54c14eb5d91751beaad7"

EVALUATION_ROWS = 32
MATH_ROWS = 16
INSTRUCTION_ROWS = 16
CANDIDATE_POOL_ROWS_PER_TASK = 50_000
DATASET_WINDOW_BLOCKS = 3_600
WEIGHT_INTERVAL_BLOCKS = 150
AUDIT_DELAY_BLOCKS = 5
AUDIT_ROWS_PER_ROUND = 8
# Eight distinct uniformly random rows are checked per round. With 32 evaluation
# rows, four successful rounds cover the complete window.
AUDIT_MIN_FAKE_ROW_FRACTION = 0.01
AUDIT_DETECTION_CONFIDENCE = 0.999
# Miner generation and validator verification share this exact rule. Among the
# top three tokens within 85% of the maximum probability, a protocol hash picks
# one canonical token. This tolerates small cross-GPU ordering changes without
# letting a miner choose whichever plausible token is convenient.
CANONICAL_MAX_CANDIDATES = 3
CANONICAL_MIN_RELATIVE_PROBABILITY = 0.85
# Validator audit memory profile. A 4B BF16 base plus one 32K sequence fits on
# a 32 GB GPU when vLLM does not reserve nearly all remaining VRAM for KV cache
# and prompt-logprob softmax is chunked below its 8K default.
AUDIT_GPU_MEMORY_UTILIZATION = 0.60
AUDIT_MAX_NUM_BATCHED_TOKENS = 4_096
AUDIT_MAX_NUM_SEQS = 8

MAX_PROMPT_CHARS = 32_768
MAX_OUTPUT_TOKENS = 32_768
MAX_CONTEXT_TOKENS = 32_768
DEFAULT_ROLLOUT_BATCH_SIZE = 16
MAX_LORA_RANK = 16
MAX_LORA_ALPHA = 256
MAX_ABS_LORA_VALUE = 100.0
MAX_ADAPTER_BYTES = 256 * 1024 * 1024
MAX_ADAPTER_ELEMENTS = 100_000_000
MAX_TENSOR_DIMENSION = 65_536
MAX_ADAPTER_CONFIG_BYTES = 64 * 1024
MAX_ROLLOUT_BYTES = 768 * 1024 * 1024
MAX_ROLLOUT_LINE_BYTES = 512 * 1024
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
        "in_proj",
        "out_proj",
        "z_proj",
        "a_proj",
        "b_proj",
    }
)
