# %% [markdown]
# ## Native TRL GSPO/GRPO Overview
# - Runs the GSPO/GRPO experiment with native Transformers, PEFT, TRL, and colocated vLLM.
# - Keeps the same dataset preparation, ordering controls, reward logic, and artifact flow
#   as the Unsloth variant.
# - Loads or resumes LoRA adapters with PEFT and trains via TRL `GRPOTrainer`.
# - Kept as the cleaner native path, though recorded Kaggle attempts hit colocated-vLLM CUDA
#   OOM.

# %% [markdown]
# ## Imports
# - Load dependencies for this notebook script.
# - Set early runtime flags before heavier stage-specific imports run.

# %%
from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")



# %% [markdown]
# ## Credentials
# - Read Kaggle, Hugging Face, and W&B credentials when available.
# - Keep committed defaults safe for public or local dry runs.

# %%
# User secrets
# try:
#     from kaggle_secrets import UserSecretsClient  # type: ignore
#
#     user_secrets = UserSecretsClient()
#     KAGGLE_KEY = user_secrets.get_secret("KAGGLE_KEY")
#     KAGGLE_USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
#     HF_KEY = user_secrets.get_secret("HF_KEY")
#     WANDB_KEY = user_secrets.get_secret("WANDB_KEY")
# except Exception:
#     KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
#     KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
#     HF_KEY = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
#     WANDB_KEY = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")

HF_KEY = WANDB_KEY = KAGGLE_KEY = KAGGLE_USERNAME = None



# %% [markdown]
# ## Kaggle Dependencies
# - Document optional Kaggle package-install commands.
# - Use cached wheels or commented commands to keep notebook startup controllable.

# %%
wheels_dir = "/kaggle/input/datasets/rohitraje0493/unsloth-vllm-wheels/packages"
# !pip install uv --no-index --find-links={wheels_dir}
# !uv pip install \
#     "triton>=3.3.0" \
#     "torchvision==0.25.0+cu128" \
#     bitsandbytes \
#     "transformers>=4.56.2" \
#     "tokenizers>=0.22.0,<=0.23.0" \
#     "trl[vllm]>=0.22.2" \
#     accelerate \
#     peft \
#     --no-index --find-links={wheels_dir}
# !uv pip install --no-deps "torchcodec==0.10.0+cu128" --no-index --find-links={wheels_dir}
# !uv pip install \
#     mamba_ssm \
#     causal_conv1d \
#     --no-index --find-links={wheels_dir}
# !uv pip install --no-deps --upgrade \
#     "torchao>=0.16.0" \
#     --no-index --find-links={wheels_dir}
# !uv pip install \
#     "math-verify[antlr4_11_0]" \
#     rapidfuzz \
#     "antlr4-python3-runtime==4.11.0" \
#     --no-index --find-links={wheels_dir}
# # !uv pip install "vllm>=0.12.0,<0.19.0" --no-index --find-links={wheels_dir}
# !uv pip install "protobuf<6.0.0" --no-index --find-links={wheels_dir}



# %% [markdown]
# ## Runtime Configuration
# - Define paths, split names, source controls, hyperparameters, and upload destinations.
# - Read values from environment variables so Kaggle and local runs can override defaults.

# %%
WORKING_DIR = Path(os.environ.get("WORKING_DIR", "/kaggle/working"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    # "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/default/1",
)
ADAPTER_INPUT_PATH = os.environ.get(
    "ADAPTER_INPUT_PATH",
    "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/lora-dpo/5",
)
BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/kaggle/input/datasets/rohitraje0493/nemotron-reasoning",
)
DATASET_REVISION = os.environ.get("DATASET_REVISION")
HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", "/tmp/hf_cache"))
TRAIN_SPLIT = os.environ.get("TRAIN_SPLIT", "train")
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", None)

def optional_nonnegative_int(
    name: str,
    default: Optional[int] = None,
) -> Optional[int]:
    value = os.environ.get(name, str(default))
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return parsed

def optional_string_list(name: str, default: Optional[str] = None) -> list[str]:
    value = os.environ.get(name, default)
    if value is None or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in value.split(",")]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError(f"{name} must be a JSON list or comma-separated strings")
    return [item.strip() for item in parsed if item.strip()]

TRAIN_MIN_IDX = optional_nonnegative_int("TRAIN_MIN_IDX")
TRAIN_MAX_IDX = optional_nonnegative_int("TRAIN_MAX_IDX", 9500)
EVAL_MIN_IDX = optional_nonnegative_int("EVAL_MIN_IDX")
EVAL_MAX_IDX = optional_nonnegative_int("EVAL_MAX_IDX")
SOURCE_OPTIONS = {
    TRAIN_SPLIT: {
        "include": optional_string_list("TRAIN_INCLUDE_SOURCES", '["nvidia-nemotron-model-reasoning-challenge", "dgxchen/nemotron-cot-tong"]'),
        "order": optional_string_list("TRAIN_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("TRAIN_EXCLUDE_SOURCES"),
        "order_remaining": os.environ.get(
            "TRAIN_ORDER_REMAINING",
            "0",
        ).lower() not in {"0", "false", "no"},
        "dpo_sort_remaining": os.environ.get(
            "TRAIN_DPO_SORT_REMAINING",
            "1",
        ).lower() not in {"0", "false", "no"},
    },
    EVAL_SPLIT: {
        "include": optional_string_list("EVAL_INCLUDE_SOURCES"),
        "order": optional_string_list("EVAL_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("EVAL_EXCLUDE_SOURCES"),
        "order_remaining": os.environ.get(
            "EVAL_ORDER_REMAINING",
            "0",
        ).lower() not in {"0", "false", "no"},
        "dpo_sort_remaining": os.environ.get(
            "EVAL_DPO_SORT_REMAINING",
            "1",
        ).lower() not in {"0", "false", "no"},
    },
}
SHUFFLE_BY_SPLIT = {
    TRAIN_SPLIT: os.environ.get("TRAIN_SHUFFLE", "0").lower()
        not in {"0", "false", "no"},
    EVAL_SPLIT: os.environ.get("EVAL_SHUFFLE", "0").lower()
        not in {"0", "false", "no"},
}
FILTER_HQ_BY_SPLIT = {
    TRAIN_SPLIT: os.environ.get("TRAIN_FILTER_HQ", "1").lower()
        not in {"0", "false", "no"},
    EVAL_SPLIT: os.environ.get("EVAL_FILTER_HQ", "1").lower()
        not in {"0", "false", "no"},
}
DPO_AWARE_BY_SPLIT = {
    TRAIN_SPLIT: os.environ.get("TRAIN_DPO_AWARE", "1").lower()
        not in {"0", "false", "no"},
    EVAL_SPLIT: os.environ.get("EVAL_DPO_AWARE", "1").lower()
        not in {"0", "false", "no"},
}

TRAIN_STAGE = os.environ.get("TRAIN_STAGE", "gspo")
TRAIN_VERSION = os.environ.get("TRAIN_VERSION", "v1")
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"nemotron-{TRAIN_STAGE}-{TRAIN_VERSION}",
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKING_DIR / RUN_NAME)))
ADAPTER_OUTPUT_DIR = Path(
    os.environ.get(
        "ADAPTER_OUTPUT_DIR",
        str(
            WORKING_DIR
            / f"nemotron-lora-{TRAIN_STAGE}-{TRAIN_VERSION}"
        ),
    )
)
HF_USERNAME = os.environ.get("HF_USERNAME", "the-submitter")
HF_ADAPTER_REPO = os.environ.get(
    "HF_ADAPTER_REPO",
    f"{HF_USERNAME}/nemotron-lora-{TRAIN_STAGE}-{TRAIN_VERSION}",
)
KAGGLE_ADAPTER_REPO = os.environ.get(
    "KAGGLE_ADAPTER_REPO",
    f"{KAGGLE_USERNAME}/nemotron-3-nano/transformers/lora-{TRAIN_STAGE}",
)
KAGGLE_DATASET_REPO = os.environ.get(
    "KAGGLE_DATASET_REPO",
    f"{KAGGLE_USERNAME}/nemotron-{TRAIN_STAGE}",
)

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "8192"))
MAX_PROMPT_LENGTH = int(os.environ.get("MAX_PROMPT_LENGTH", "4096"))
MAX_COMPLETION_LENGTH = int(
    os.environ.get(
        "MAX_COMPLETION_LENGTH",
        "7680",
        # str(MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH),
    )
)
if MAX_PROMPT_LENGTH <= 0 or MAX_COMPLETION_LENGTH <= 0:
    raise ValueError("Prompt and completion lengths must be positive")

DATASET_WORKERS = max(1, int(os.environ.get("DATASET_NUM_PROC", "8")))
DATASET_NUM_PROC = DATASET_WORKERS if DATASET_WORKERS > 1 else None
SEED = int(os.environ.get("SEED", "3407"))

PER_DEVICE_TRAIN_BATCH_SIZE = int(
    os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "2")
)
PER_DEVICE_EVAL_BATCH_SIZE = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")
)
GENERATION_BATCH_SIZE = optional_nonnegative_int("GENERATION_BATCH_SIZE")
STEPS_PER_GENERATION = optional_nonnegative_int("STEPS_PER_GENERATION")
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-6"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "1"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "5"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(SAVE_STEPS)))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))
LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
TOP_K = optional_nonnegative_int("TOP_K")
GRPO_BETA = float(os.environ.get("GRPO_BETA", "0.0"))
GRPO_LOSS_TYPE = os.environ.get("GRPO_LOSS_TYPE", "dr_grpo")
VLLM_GPU_MEMORY_UTILIZATION = float(
    os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.95")
)
QUANTIZATION = os.environ.get("QUANTIZATION", "1").lower() not in {"0", "false", "no"}
MATH_VERIFY_TIMEOUT_SECONDS = int(
    os.environ.get("MATH_VERIFY_TIMEOUT_SECONDS", "5")
)
if MATH_VERIFY_TIMEOUT_SECONDS <= 0:
    raise ValueError("MATH_VERIFY_TIMEOUT_SECONDS must be positive")

EXACT_MATCH_WEIGHT = float(os.environ.get("EXACT_MATCH_WEIGHT", "5.0"))
ANSWER_FUZZY_WEIGHT = float(os.environ.get("ANSWER_FUZZY_WEIGHT", "3.0"))
COMPLETION_FUZZY_WEIGHT = float(
    os.environ.get("COMPLETION_FUZZY_WEIGHT", "0.15")
)
BOXED_WEIGHT = float(os.environ.get("BOXED_WEIGHT", "1.0"))
THINK_WEIGHT = float(os.environ.get("THINK_WEIGHT", "0.25"))

REPORT_TO = os.environ.get("REPORT_TO", "wandb")
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "0")
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "0").lower() not in {
    "0",
    "false",
    "no",
}
PUSH_TO_KAGGLE = os.environ.get("PUSH_TO_KAGGLE", "0").lower() not in {
    "0",
    "false",
    "no",
}
KEEP_IN_MEMORY = os.environ.get("KEEP_IN_MEMORY", "1").lower() not in {
    "0",
    "false",
    "no",
}

BOXED_ANSWER_INSTRUCTION = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
BOXED_START_RE = re.compile(r"\\boxed\{")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
FALLBACK_ANSWER_PATTERNS = [
    re.compile(r"The final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
]
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
BINARY_RE = re.compile(r"[01]+")

if REPORT_TO == "wandb":
    os.environ.setdefault("WANDB_MODE", "offline")
    wandb_dir = Path(os.environ.get("WANDB_DIR", str(WORKING_DIR / "wandb_logs")))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ.setdefault("WANDB_SILENT", "true")




# %% [markdown]
# ## Helper Functions
# - Define reusable helpers including `clean_text`, `is_high_quality_example`,
#   `is_reward_example`, `build_user_content`, `build_grpo_example`, `dpo_priority`.
# - Support parsing, filtering, formatting, verification, loading, or upload behavior used
#   later.

# %%
def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


HQ_SOURCES = {
    "nvidia-nemotron-model-reasoning-challenge",
    "dgxchen/nemotron-cot-tong",
}

HQ_ANSWER_TYPES = {"integer", "float", "fraction"}

def is_high_quality_example(example: dict[str, Any]) -> bool:
    if example.get("response") and not example.get("reasoning"):
        return False
    if example.get("source") in HQ_SOURCES:
        return True
    answer_type = clean_text(example.get("answer_type"))
    if answer_type is not None and answer_type.lower() in HQ_ANSWER_TYPES:
        return True
    return False
    # final_answer = clean_text(example.get("final_answer"))
    # return final_answer is not None and final_answer.isalnum()


def is_reward_example(example: dict[str, Any]) -> bool:
    return (
        clean_text(example.get("prompt")) is not None
        and clean_text(example.get("final_answer")) is not None
    )


def build_user_content(prompt: Any) -> str:
    normalized_prompt = clean_text(prompt)
    if normalized_prompt is None:
        raise ValueError("Cannot format GRPO data without a prompt")
    return normalized_prompt + BOXED_ANSWER_INSTRUCTION


def build_grpo_example(example: dict[str, Any]) -> dict[str, Any]:
    final_answer = clean_text(example.get("final_answer"))
    if final_answer is None:
        raise ValueError("Dataset must be filtered before GRPO formatting")
    return {
        "prompt": [
            {
                "role": "user",
                "content": build_user_content(example.get("prompt")),
            }
        ],
        "response": clean_text(example.get("response")),
        "reasoning": clean_text(example.get("reasoning")),
        "final_answer": final_answer,
    }


def dpo_priority(example: dict[str, Any]) -> int:
    chosen = clean_text(example.get("chosen"))
    rejected = clean_text(example.get("rejected"))
    selected = bool(example.get("dpo_selected"))
    if rejected is not None and chosen is None:
        return 0
    if selected and chosen is not None and rejected is not None:
        return 1
    if selected:
        return 2
    return 3




# %% [markdown]
# ## Dataset Loading
# - Load datasets from local disk, Kaggle-mounted paths, Hugging Face repos, parquet files,
#   or saved DatasetDicts.
# - Normalize split handling and cache behavior for downstream processing.

# %%
def load_reasoning_dataset():
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    dataset_path = Path(DATASET_PATH)
    if dataset_path.exists():
        if (
            (dataset_path / "dataset_dict.json").exists()
            or (dataset_path / "dataset_info.json").exists()
        ):
            loaded = load_from_disk(
                str(dataset_path),
                keep_in_memory=KEEP_IN_MEMORY,
            )
        else:
            parquet_files = sorted(dataset_path.rglob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"No Hugging Face dataset or parquet files found at {dataset_path}"
                )
            loaded = load_dataset(
                "parquet",
                data_dir=str(dataset_path),
                cache_dir=str(HF_CACHE_DIR),
                keep_in_memory=KEEP_IN_MEMORY,
            )
    else:
        loaded = load_dataset(
            DATASET_PATH,
            revision=DATASET_REVISION,
            token=HF_KEY,
            cache_dir=str(HF_CACHE_DIR),
            keep_in_memory=KEEP_IN_MEMORY,
        )

    if isinstance(loaded, Dataset):
        loaded = DatasetDict({TRAIN_SPLIT: loaded})
    if TRAIN_SPLIT not in loaded:
        raise KeyError(
            f"Training split {TRAIN_SPLIT!r} is unavailable; "
            f"found {list(loaded.keys())}"
        )
    return loaded


def apply_source_options(dataset, split_name: str):
    options = SOURCE_OPTIONS.get(
        split_name,
        {"include": [], "order": [], "exclude": []},
    )
    include_sources = options["include"]
    order_sources = options["order"]
    exclude_sources = set(options["exclude"])
    order_remaining = options.get("order_remaining", False)
    dpo_sort_remaining = options.get("dpo_sort_remaining", True)
    dpo_aware = DPO_AWARE_BY_SPLIT.get(split_name)
    if (
        not include_sources
        and not order_sources
        and not exclude_sources
        and not dpo_aware
    ):
        return dataset
    if "source" not in dataset.column_names:
        raise KeyError(
            f"{split_name}: source controls require a 'source' column"
        )

    if exclude_sources:
        dataset = dataset.filter(
            lambda example: example.get("source") not in exclude_sources,
            num_proc=DATASET_NUM_PROC,
            desc=f"{split_name}: exclude sources",
            keep_in_memory=KEEP_IN_MEMORY,
        )

    available_sources = list(dict.fromkeys(dataset["source"]))
    missing_includes = [
        source for source in include_sources if source not in available_sources
    ]
    if missing_includes:
        raise ValueError(
            f"{split_name}: required INCLUDE_SOURCES are unavailable: "
            f"{missing_includes}"
        )

    include_sources = list(dict.fromkeys(include_sources))
    explicit_order = [
        source
        for source in order_sources
        if source != "*" and source not in include_sources
    ]
    remaining_sources = [
        source
        for source in available_sources
        if source not in include_sources and source not in explicit_order
    ]
    if "*" in order_sources:
        wildcard_index = order_sources.index("*")
        before_wildcard = {
            source for source in order_sources[:wildcard_index] if source != "*"
        }
        before_remaining_sources = (
            include_sources
            + [source for source in explicit_order if source in before_wildcard]
        )
        after_remaining_sources = [
            source for source in explicit_order if source not in before_wildcard
        ]
    else:
        before_remaining_sources = include_sources + explicit_order
        after_remaining_sources = []
    ordered_sources = (
        before_remaining_sources
        + (remaining_sources if order_remaining else ["*"])
        + after_remaining_sources
    )

    indices_by_source: dict[Any, list[int]] = {}
    for index, source in enumerate(dataset["source"]):
        indices_by_source.setdefault(source, []).append(index)

    def ordered_source_indices(source: Any) -> list[int]:
        source_indices = indices_by_source.get(source, [])
        if dpo_aware:
            source_indices = sorted(
                source_indices,
                key=lambda index: dpo_priority(dataset[index]),
            )
        return source_indices

    before_remaining_indices = [
        index
        for source in before_remaining_sources
        for index in ordered_source_indices(source)
    ]
    if order_remaining:
        remaining_indices = [
            index
            for source in remaining_sources
            for index in ordered_source_indices(source)
        ]
    else:
        remaining_source_set = set(remaining_sources)
        if dpo_aware and dpo_sort_remaining:
            remaining_indices_by_source = {
                source: iter(ordered_source_indices(source))
                for source in remaining_sources
            }
            remaining_indices = [
                next(remaining_indices_by_source[source])
                for source in dataset["source"]
                if source in remaining_source_set
            ]
        else:
            remaining_indices = [
                index
                for index, source in enumerate(dataset["source"])
                if source in remaining_source_set
            ]
    after_remaining_indices = [
        index
        for source in after_remaining_sources
        for index in ordered_source_indices(source)
    ]
    selected_indices = (
        before_remaining_indices
        + remaining_indices
        + after_remaining_indices
    )

    ordered = dataset.select(selected_indices, keep_in_memory=KEEP_IN_MEMORY)
    print(
        f"{split_name}: source order={ordered_sources}; "
        f"order_remaining={order_remaining}; dpo_aware={dpo_aware}; "
        f"dpo_sort_remaining={dpo_sort_remaining}; "
        f"retained={len(ordered):,}"
    )
    return ordered


def prepare_split(
    dataset,
    split_name: str,
    allow_empty: bool = False,
):
    from datasets import Features, List, Value

    original_size = len(dataset)
    dataset = dataset.filter(
        is_reward_example,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: keep reward-ready examples",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    if FILTER_HQ_BY_SPLIT.get(split_name, False):
        before_hq = len(dataset)
        dataset = dataset.filter(
            is_high_quality_example,
            num_proc=DATASET_NUM_PROC,
            desc=f"{split_name}: keep high-quality examples",
            keep_in_memory=KEEP_IN_MEMORY,
        )
        print(
            f"{split_name}: HQ filter retained "
            f"{len(dataset):,}/{before_hq:,} examples"
        )
    if not len(dataset):
        if allow_empty:
            print(f"{split_name}: no reward-ready examples after filtering")
            return None
        raise ValueError(f"{split_name} has no reward-ready examples")

    grpo_features = Features(
        {
            "prompt": List(
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "reasoning_content": Value("string"),
                }
            ),
            "response": Value("string"),
            "reasoning": Value("string"),
            "final_answer": Value("string"),
        }
    )
    dataset = dataset.map(
        build_grpo_example,
        remove_columns=dataset.column_names,
        features=grpo_features,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: build GRPO conversations",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    print(f"{split_name}: retained {len(dataset):,}/{original_size:,} examples")
    return dataset


def select_index_range(
    dataset,
    min_idx: Optional[int],
    max_idx: Optional[int],
    split_name: str,
    shuffle: bool,
):
    start = 0 if min_idx is None else min_idx
    stop = len(dataset) if max_idx is None else min(max_idx, len(dataset))
    if start > stop:
        raise ValueError(
            f"{split_name}: min index {start} exceeds max index {stop}"
        )
    if start >= len(dataset):
        raise ValueError(
            f"{split_name}: min index {start} is outside dataset size {len(dataset)}"
        )
    selected = (
        dataset
        if start == 0 and stop == len(dataset)
        else dataset.select(
            range(start, stop),
            keep_in_memory=KEEP_IN_MEMORY,
        )
    )
    if not len(selected):
        raise ValueError(
            f"{split_name}: index range [{start}, {stop}) selected no examples"
        )
    if min_idx is not None or max_idx is not None:
        print(
            f"{split_name}: selected [{start:,}, {stop:,}) "
            f"({len(selected):,} examples)"
        )
    if shuffle:
        selected = selected.shuffle(
            seed=SEED,
            keep_in_memory=KEEP_IN_MEMORY,
        )
        print(f"{split_name}: shuffled {len(selected):,} examples with seed {SEED}")
    return selected


def prepare_datasets():
    dataset_dict = load_reasoning_dataset()
    train_source_dataset = apply_source_options(
        dataset_dict[TRAIN_SPLIT],
        TRAIN_SPLIT,
    )
    train_dataset = prepare_split(
        train_source_dataset,
        TRAIN_SPLIT,
    )
    train_dataset = select_index_range(
        train_dataset,
        TRAIN_MIN_IDX,
        TRAIN_MAX_IDX,
        TRAIN_SPLIT,
        SHUFFLE_BY_SPLIT[TRAIN_SPLIT],
    )

    eval_dataset = None
    if (
        EVAL_SPLIT
        and EVAL_SPLIT in dataset_dict
        and len(dataset_dict[EVAL_SPLIT])
    ):
        eval_source_dataset = apply_source_options(
            dataset_dict[EVAL_SPLIT],
            EVAL_SPLIT,
        )
        eval_dataset = prepare_split(
            eval_source_dataset,
            EVAL_SPLIT,
            allow_empty=True,
        )
        if eval_dataset is not None:
            eval_dataset = select_index_range(
                eval_dataset,
                EVAL_MIN_IDX,
                EVAL_MAX_IDX,
                EVAL_SPLIT,
                SHUFFLE_BY_SPLIT[EVAL_SPLIT],
            )
    return train_dataset, eval_dataset




# %% [markdown]
# ## Answer Extraction and Rewards
# - Extract boxed or fallback final answers and verify them against references.
# - Compute reward components for exactness, fuzzy similarity, boxed formatting, and
#   reasoning tags.

# %%
import math_verify
from rapidfuzz import fuzz, utils

def extract_competition_boxed_answer(text: Any) -> Optional[str]:
    if not text:
        return None
    value = str(text)
    boxed_starts = list(BOXED_START_RE.finditer(value))
    matches = []
    for index, match in enumerate(boxed_starts):
        start = match.end()
        end = (
            boxed_starts[index + 1].start()
            if index + 1 < len(boxed_starts)
            else len(value)
        )
        segment = value[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if not matches:
        return None
    non_empty = [match.strip() for match in matches if match.strip()]
    return non_empty[-1] if non_empty else matches[-1].strip()


def extract_boxed_spans(text: Any) -> list[tuple[int, int, str]]:
    if not text:
        return []
    value = str(text)
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    marker = r"\boxed{"
    while True:
        start = value.find(marker, cursor)
        if start < 0:
            break
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(value) and depth:
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            spans.append((start, index, value[content_start : index - 1]))
            cursor = index
        else:
            cursor = content_start
    return spans


def extract_balanced_boxed_answer(text: Any) -> Optional[str]:
    spans = extract_boxed_spans(text)
    if not spans:
        return None
    non_empty = [
        answer.strip()
        for _start, _end, answer in spans
        if answer.strip()
    ]
    return non_empty[-1] if non_empty else spans[-1][2].strip()


def extract_fallback_answer(text: Any) -> Optional[str]:
    value = clean_text(text)
    if value is None:
        return None
    for pattern in FALLBACK_ANSWER_PATTERNS:
        matches = pattern.findall(value)
        if matches:
            return matches[-1].strip()
    matches = NUMBER_RE.findall(value)
    if matches:
        return matches[-1]
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1] if lines else None


def extract_final_answers(text: Any) -> list[Optional[str]] | Optional[str]:
    boxed_answers = [
        extract_competition_boxed_answer(text),
        extract_balanced_boxed_answer(text),
    ]
    if any(clean_text(answer) is not None for answer in boxed_answers):
        return boxed_answers
    fallback_answer = extract_fallback_answer(text)
    return fallback_answer


def verify(stored_answer: Any, predicted: Any) -> bool:
    stored = clean_text(stored_answer)
    prediction = clean_text(predicted)
    if not stored:
        return not prediction

    if BINARY_RE.fullmatch(stored):
        return prediction.casefold() == stored.casefold()

    try:
        if math.isclose(
            float(stored),
            float(prediction),
            rel_tol=1e-2,
            abs_tol=1e-5,
        ):
            return True
    except Exception:
        pass

    try:
        if math_verify.verify(
            math_verify.parse(stored),
            math_verify.parse(prediction),
            float_rounding=2,
            numeric_precision=2,
            strict=True,
            allow_set_relation_comp=True,
            timeout_seconds=MATH_VERIFY_TIMEOUT_SECONDS,
        ):
            return True
    except Exception:
        pass

    return prediction.casefold() == stored.casefold()


def combine_reasoning_response(reasoning: Any, response: Any) -> Optional[str]:
    normalized_response = clean_text(response)
    if normalized_response is None:
        return None
    normalized_reasoning = clean_text(reasoning)
    if normalized_reasoning is None:
        return normalized_response
    return (
        f"<think>\n{normalized_reasoning}\n</think>\n"
        f"{normalized_response}"
    )


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    # if isinstance(completion, list):
    #     return "".join(
    #         str(message.get("content") or "")
    #         for message in completion
    #         if isinstance(message, dict)
    #     )
    try:
        return str(completion[0].get("content", ""))
    except Exception:
        pass
    return str(completion or "")


def normalized_fuzzy_score(gold: Any, target: Any) -> float:
    if gold is None:
        return 0.0
    gold_text = utils.default_process(clean_text(gold) or "")
    target_text = utils.default_process(clean_text(target) or "")
    return fuzz.ratio(gold_text, target_text) / 100.0


def normalized_token_set_score(gold: Any, target: Any) -> float:
    if gold is None:
        return 0.0
    gold_text = utils.default_process(clean_text(gold) or "")
    target_text = utils.default_process(clean_text(target) or "")
    return fuzz.token_set_ratio(gold_text, target_text) / 100.0


def unified_reward(
    prompts,
    completions,
    response,
    reasoning,
    final_answer,
    **kwargs,
) -> list[float]:
    scores: list[float] = []

    for completion, reference_response, reference_reasoning, target in zip(
        completions,
        response,
        reasoning,
        final_answer,
        strict=True,
    ):
        text = completion_text(completion)
        normalized_text = text.casefold()
        if "</think>" in normalized_text and "<think>" not in normalized_text:
            text = f"<think>\n{text}"

        extracted_answers = extract_final_answers(text)

        if isinstance(extracted_answers, list):
            boxed_answers = extracted_answers
        else:
            extracted_answers = [extracted_answers]
            boxed_answers = [None]

        exact_score = max(
            (
                1.0 if verify(target, extracted_answer) else 0.0
                for extracted_answer in extracted_answers
            ),
            default=0.0,
        )
        answer_fuzzy_score = max(
            (
                normalized_fuzzy_score(target, extracted_answer)
                for extracted_answer in extracted_answers
            ),
            default=0.0,
        )
        boxed_score = max(
            (
                1.0 if clean_text(extracted_answer) is not None else 0.0
                for extracted_answer in boxed_answers
            ),
            default=0.0,
        )
        reference_completion = combine_reasoning_response(
            reference_reasoning,
            reference_response,
        )
        completion_fuzzy_score = normalized_token_set_score(
            reference_completion,
            text,
        )
        think_matches = [
            match.group(1).strip()
            for match in THINK_RE.finditer(text)
        ]
        think_score = 1.0 if any(think_matches) else 0.0

        scores.append(
            EXACT_MATCH_WEIGHT * exact_score
            + ANSWER_FUZZY_WEIGHT * answer_fuzzy_score
            + COMPLETION_FUZZY_WEIGHT * completion_fuzzy_score
            + BOXED_WEIGHT * boxed_score
            + THINK_WEIGHT * think_score
        )
    return scores




# %% [markdown]
# ## Model and Adapter Loading
# - Load the base model, tokenizer, and optional LoRA adapter.
# - Configure trainable adapter parameters and runtime compatibility settings.

# %%
def prepare_adapter_input_path() -> Optional[str]:
    if ADAPTER_INPUT_PATH is None:
        return None

    source_path = Path(ADAPTER_INPUT_PATH)
    if not (source_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"adapter_config.json does not exist under {source_path}"
        )

    adapter_path = source_path
    if str(source_path).startswith("/kaggle/input"):
        adapter_path = WORKING_DIR / "adapter_input"
        if adapter_path.exists():
            shutil.rmtree(adapter_path)
        shutil.copytree(source_path, adapter_path)

    if BASE_MODEL_ID and MODEL_PATH:
        readme_path = adapter_path / "README.md"
        if readme_path.exists():
            readme_path.write_text(
                readme_path.read_text().replace(BASE_MODEL_ID, MODEL_PATH)
            )

        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            config_path.write_text(
                config_path.read_text().replace(BASE_MODEL_ID, MODEL_PATH)
            )

    return str(adapter_path)


def load_model_and_tokenizer():
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_input_path = prepare_adapter_input_path()
    tokenizer_source = (
        adapter_input_path
        if adapter_input_path is not None
        and (Path(adapter_input_path) / "tokenizer_config.json").exists()
        else MODEL_PATH
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        token=HF_KEY,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    bnb_config = None
    if QUANTIZATION:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=model_dtype,
        trust_remote_code=True,
        token=HF_KEY,
        quantization_config=bnb_config,
        # attn_implementation=os.environ.get(
        #     "ATTN_IMPLEMENTATION",
        #     "eager",
        # ),
        low_cpu_mem_usage=False,
        device_map="auto",
    )
    model.config.use_cache = False

    if adapter_input_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_input_path,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=LORA_R,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "in_proj",
                "out_proj",
            ],
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_rslora=False,
        )
        model = get_peft_model(model, lora_config)

    model.name_or_path = MODEL_PATH
    model.config._name_or_path = MODEL_PATH
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters == 0:
        raise RuntimeError("GRPO/GSPO model has no trainable parameters")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Trainable adapter parameters: {trainable_parameters:,}/"
        f"{total_parameters:,} "
        f"({100 * trainable_parameters / total_parameters:.4f}%)"
    )
    return model, tokenizer


def resolve_resume_from_checkpoint() -> bool | str | None:
    if RESUME_FROM_CHECKPOINT is None:
        return None
    normalized = RESUME_FROM_CHECKPOINT.strip()
    if normalized.lower() in {"1", "true", "yes"}:
        return True
    if normalized.lower() in {"0", "false", "no", ""}:
        return None
    return normalized




# %% [markdown]
# ## Training Runtime Bootstrap
# - Initialize tokenizer runtime settings and output directories.
# - Import Torch and TRL GRPO utilities, then verify CUDA availability.
# %%
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import torch
from trl import GRPOConfig, GRPOTrainer

if not torch.cuda.is_available():
    raise RuntimeError("GRPO/GSPO training requires a CUDA GPU")



# %% [markdown]
# ## Stage Preparation
# - Materialize the model, tokenizer, dataset splits, or inference engine needed by later
#   cells.
# - Keep heavyweight setup isolated before training or generation begins.

# %%
model, tokenizer = load_model_and_tokenizer()



# %% [markdown]
# ## Stage Preparation
# - Materialize the model, tokenizer, dataset splits, or inference engine needed by later
#   cells.
# - Keep heavyweight setup isolated before training or generation begins.

# %%
train_dataset, eval_dataset = prepare_datasets()



# %% [markdown]
# ## Helper Functions
# - Define reusable helpers including `overwrite_engine_args`.
# - Support parsing, filtering, formatting, verification, loading, or upload behavior used
#   later.

# %%
# TRL VLLM Patch

import inspect
import functools
from vllm.engine.arg_utils import EngineArgs, AsyncEngineArgs
from vllm import LLM

def overwrite_engine_args(original_func):
    # Capture the original function's true signature object
    original_sig = inspect.signature(original_func)
    
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        kwargs["trust_remote_code"] = True
        kwargs["max_model_len"] = MAX_SEQ_LENGTH
        kwargs["enable_prefix_caching"] = True
        kwargs["enable_chunked_prefill"] = True
        if QUANTIZATION:
            kwargs["quantization"] = "bitsandbytes"
        # kwargs = dict(
        #     model=kwargs["model"],
        #     tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
        #     max_num_seqs=kwargs.get("max_num_seqs", 64),
        #     gpu_memory_utilization=kwargs.get("gpu_memory_utilization", VLLM_GPU_MEMORY_UTILIZATION),
        #     dtype="auto",
        #     max_model_len=kwargs.get("max_model_len", MAX_SEQ_LENGTH),
        #     trust_remote_code=True,
        #     enable_lora=kwargs.get("enable_lora", True),
        #     max_lora_rank=kwargs.get("max_lora_rank", MAX_LORA_RANK),
        #     enable_prefix_caching=kwargs.get("enable_prefix_caching", True),
        #     enable_chunked_prefill=kwargs.get("enable_chunked_prefill", True),
        #     max_logprobs=kwargs.get("max_logprobs", 0),
        #     seed=kwargs.get("seed", SEED),
        #     disable_log_stats=kwargs.get("disable_log_stats", True),
        #     enforce_eager=kwargs.get("enforce_eager", False),
        #     enable_sleep_mode=kwargs.get("enable_sleep_mode", True),
        # )
        return original_func(*args, **kwargs)
    
    # Explicitly attach the original signature to pass 'inspect' validation
    wrapper.__signature__ = original_sig
    return wrapper

# Apply to all entrypoints
EngineArgs.__init__ = overwrite_engine_args(EngineArgs.__init__)
AsyncEngineArgs.__init__ = overwrite_engine_args(AsyncEngineArgs.__init__)
LLM.__init__ = overwrite_engine_args(LLM.__init__)



# %% [markdown]
# ## Trainer Configuration
# - Configure trainer-specific hyperparameters, precision, checkpointing, evaluation, and
#   reporting.
# - Bind optimizer, batching, sequence length, and stage-specific training options.

# %%
has_eval = eval_dataset is not None and len(eval_dataset) > 0
bf16 = torch.cuda.is_bf16_supported()

training_args = GRPOConfig(
    output_dir=str(OUTPUT_DIR),
    run_name=RUN_NAME,
    max_completion_length=MAX_COMPLETION_LENGTH,
    # generation_kwargs={"max_length": MAX_SEQ_LENGTH},
    num_generations=NUM_GENERATIONS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    vllm_enable_sleep_mode=True,
    importance_sampling_level="sequence",
    beta=GRPO_BETA,
    loss_type=GRPO_LOSS_TYPE,
    scale_rewards=False,
    mask_truncated_completions=True,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    generation_batch_size=GENERATION_BATCH_SIZE,
    steps_per_generation=STEPS_PER_GENERATION,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    eval_strategy="steps" if has_eval else "no",
    eval_steps=EVAL_STEPS if has_eval else None,
    eval_accumulation_steps=GRADIENT_ACCUMULATION_STEPS if has_eval else None,
    load_best_model_at_end=has_eval,
    metric_for_best_model="eval_reward" if has_eval else None,
    greater_is_better=True if has_eval else None,
    adam_beta1=0.9,
    adam_beta2=0.99,
    adam_epsilon=1e-8,
    optim="adamw_8bit",
    epsilon=3e-4,
    epsilon_high=4e-4,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    max_grad_norm=0.1,
    top_entropy_quantile=1.0,
    seed=SEED,
    data_seed=SEED,
    report_to=REPORT_TO,
    bf16=bf16,
    fp16=not bf16,
    tf32=None,
    remove_unused_columns=False,
    shuffle_dataset=SHUFFLE_BY_SPLIT[TRAIN_SPLIT],
    use_transformers_paged=False,
    cache_implementation=None,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[unified_reward],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)



# %% [markdown]
# ## Runtime Sanity Checks
# - Print representative samples, reward values, GPU details, or runtime metrics.
# - Help catch formatting and resource issues before or after long-running work.

# %%
print("GRPO/GSPO training sample:")
print(train_dataset[0])
sample_reward = unified_reward(
    prompts=[train_dataset[0]["prompt"]],
    completions=[
        [
            {
                "role": "assistant",
                "content": (
                    "<think>\nExample reasoning\n</think>\n"
                    f"\\boxed{{{train_dataset[0]['final_answer']}}}"
                ),
            }
        ]
    ],
    response=[train_dataset[0]["response"]],
    reasoning=[train_dataset[0]["reasoning"]],
    final_answer=[train_dataset[0]["final_answer"]],
)
print(f"Reward sanity check: {sample_reward}")

gpu = torch.cuda.get_device_properties(0)
start_reserved = torch.cuda.max_memory_reserved() / 1024**3
print(
    f"GPU: {gpu.name}; VRAM={gpu.total_memory / 1024**3:.2f} GiB; "
    f"initial_reserved={start_reserved:.2f} GiB"
)



# %% [markdown]
# ## Training Execution
# - Start or resume training from the configured checkpoint.
# - Collect training statistics for later runtime and memory reporting.

# %%
trainer_stats = trainer.train(
    resume_from_checkpoint=resolve_resume_from_checkpoint()
)



# %% [markdown]
# ## Save Artifacts
# - Persist datasets, adapters, tokenizer files, trainer state, or output folders.
# - Prepare generated artifacts for reuse and optional publishing.

# %%
trainer.save_state()
model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))



# %% [markdown]
# ## Adapter Metadata Normalization
# - Rewrite saved adapter metadata back from Kaggle-local model paths to the original base model id.
# - Keep uploaded adapter configs portable outside the Kaggle runtime.
# %%
if MODEL_PATH and BASE_MODEL_ID:
    readme_path = ADAPTER_OUTPUT_DIR / "README.md"
    if readme_path.exists():
        readme_path.write_text(
            readme_path.read_text().replace(MODEL_PATH, BASE_MODEL_ID)
        )
    config_path = ADAPTER_OUTPUT_DIR / "adapter_config.json"
    if config_path.exists():
        config_path.write_text(
            config_path.read_text().replace(MODEL_PATH, BASE_MODEL_ID)
        )



# %% [markdown]
# ## Runtime Sanity Checks
# - Print representative samples, reward values, GPU details, or runtime metrics.
# - Help catch formatting and resource issues before or after long-running work.

# %%
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
runtime = trainer_stats.metrics.get("train_runtime", 0.0)
print(f"Training runtime: {runtime:.2f} seconds ({runtime / 60:.2f} minutes)")
print(f"Peak reserved VRAM: {peak_reserved:.2f} GiB")
print(f"Saved GRPO/GSPO LoRA adapter to {ADAPTER_OUTPUT_DIR}")



# %% [markdown]
# ## Upload Artifacts
# - Publish generated datasets or model adapters when upload flags are enabled.
# - Use Hugging Face and Kaggle APIs with configured credentials.

# %%
if PUSH_TO_HUB:
    try:
        if not HF_KEY:
            raise RuntimeError("PUSH_TO_HUB=1 but HF_KEY/HF_TOKEN is not configured")
        model.push_to_hub(
            HF_ADAPTER_REPO,
            token=HF_KEY,
            private=True,
        )
        tokenizer.push_to_hub(
            HF_ADAPTER_REPO,
            token=HF_KEY,
            private=True,
        )
    except Exception as e:
        try:
            from huggingface_hub import HfApi, login

            login(token=HF_KEY)
            api = HfApi()

            api.create_repo(
                repo_id=HF_ADAPTER_REPO,
                repo_type="model",
                private=True,
                exist_ok=True
            )

            api.upload_folder(
                folder_path=ADAPTER_OUTPUT_DIR,
                repo_id=HF_ADAPTER_REPO,
                repo_type="model"
            )
        except Exception as exc:
            print(f"Upload to Hugging Face failed: {e}, {exc}")
        else:
            print(f"Upload to Hugging Face succeeded")
    else:
        print(f"Upload to Hugging Face succeeded")



# %% [markdown]
# ## Upload Artifacts
# - Publish generated datasets or model adapters when upload flags are enabled.
# - Use Hugging Face and Kaggle APIs with configured credentials.

# %%
if PUSH_TO_KAGGLE:
    try:
        if not KAGGLE_USERNAME or not KAGGLE_KEY:
            raise RuntimeError(
                "PUSH_TO_KAGGLE=1 but KAGGLE credentials are not configured"
            )
        import kagglehub

        kagglehub.model_upload(
            handle=KAGGLE_ADAPTER_REPO,
            local_model_dir=str(ADAPTER_OUTPUT_DIR),
            version_notes=f"Nemotron LoRA continued from sft to {TRAIN_STAGE}",
            license_name="Apache 2.0",
        )

        (WORKING_DIR / "state.db").unlink(missing_ok=True)
        kagglehub.dataset_upload(
            handle=KAGGLE_DATASET_REPO,
            local_dataset_dir=WORKING_DIR,
            version_notes=f"nemotron {TRAIN_STAGE}",
        )
    except Exception as error:
        print(f"Upload to Kaggle failed: {error}")
    else:
        print("Upload to Kaggle succeeded")
