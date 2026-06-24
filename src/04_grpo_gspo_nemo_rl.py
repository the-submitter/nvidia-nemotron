# %% [markdown]
# ## NeMo-RL GSPO/GRPO Overview
# - Runs GSPO/GRPO-style training through NeMo-RL with JSONL response data, a custom
#   reward environment, and colocated vLLM generation.
# - Builds reward-ready prompts with source, HQ, DPO-aware, and remaining-source ordering
#   controls.
# - Combines exact answer verification, fuzzy matching, completion similarity, boxed
#   formatting, and think-tag rewards.
# - Keeps the existing dataset preparation and reward logic while avoiding Unsloth/vLLM
#   synchronization issues on the Nemotron hybrid model.

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
wheels_dir = "/kaggle/input/datasets/rohitraje0493/nemo-rl-vllm-wheels/packages"
# !pip install uv --no-index --find-links={wheels_dir}
# !uv pip install \
#     "vllm>=0.12.0,<0.19.0" \
#     "transformers>=4.56.2,<5.0.0" \
#     "tokenizers>=0.22.0,<=0.23.0" \
#     "math-verify[antlr4_11_0]" \
#     rapidfuzz \
#     "antlr4-python3-runtime==4.11.0" \
#     "protobuf<6.0.0" \
#     --no-index --find-links={wheels_dir}
# If cached wheels are unavailable, install NeMo-RL from NVIDIA-NeMo/RL in a separate setup cell.



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
MAX_LORA_RANK = int(os.environ.get("MAX_LORA_RANK", "32"))

NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
TOP_K = optional_nonnegative_int("TOP_K")
GRPO_BETA = float(os.environ.get("GRPO_BETA", "0.0"))
GRPO_LOSS_TYPE = os.environ.get("GRPO_LOSS_TYPE", "dr_grpo")
VLLM_GPU_MEMORY_UTILIZATION = float(
    os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.95")
)
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
# ## Adapter Input Preparation
# - Resolve an optional PEFT adapter path from a Kaggle input or local folder.
# - Normalize adapter metadata when a Kaggle-local base model path replaces the original model id.

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
# ## NeMo-RL JSONL Settings
# - Configure response-dataset JSONL locations consumed by NeMo-RL.
# - Keep backend-specific knobs isolated from the common dataset and reward code.

# %%
NEMO_DATA_DIR = Path(os.environ.get("NEMO_DATA_DIR", WORKING_DIR / "nemo_rl_data"))
NEMO_TRAIN_JSONL = NEMO_DATA_DIR / "train.jsonl"
NEMO_EVAL_JSONL = NEMO_DATA_DIR / "validation.jsonl"
NEMO_BASE_CONFIG = os.environ.get(
    "NEMO_BASE_CONFIG",
    "examples/configs/grpo_math_1B.yaml",
)
NEMO_TASK_NAME = os.environ.get("NEMO_TASK_NAME", "nemotron_reasoning")
NEMO_PROCESSOR_NAME = os.environ.get(
    "NEMO_PROCESSOR_NAME",
    "nemotron_grpo_data_processor",
)
NEMO_ENV_NAME = os.environ.get("NEMO_ENV_NAME", "nemotron_unified_reward")
NEMO_ENV_FQN = os.environ.get(
    "NEMO_ENV_FQN",
    f"{__name__}.NemotronUnifiedRewardEnvironment",
)
NEMO_ENV_NUM_WORKERS = int(os.environ.get("NEMO_ENV_NUM_WORKERS", "1"))
NEMO_TRAIN_GLOBAL_BATCH_SIZE = int(
    os.environ.get(
        "NEMO_TRAIN_GLOBAL_BATCH_SIZE",
        str(max(PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, 1)),
    )
)
NEMO_GENERATION_BATCH_SIZE = int(
    os.environ.get("NEMO_GENERATION_BATCH_SIZE", str(GENERATION_BATCH_SIZE))
)
NEMO_TENSOR_PARALLEL_SIZE = int(os.environ.get("NEMO_TENSOR_PARALLEL_SIZE", "1"))
NEMO_PIPELINE_PARALLEL_SIZE = int(os.environ.get("NEMO_PIPELINE_PARALLEL_SIZE", "1"))
NEMO_SEQUENCE_LEVEL_IS = bool_env("NEMO_SEQUENCE_LEVEL_IS", True)
NEMO_NORMALIZE_REWARDS = bool_env("NEMO_NORMALIZE_REWARDS", False)
NEMO_RUN_TRAIN = bool_env("NEMO_RUN_TRAIN", True)




# %% [markdown]
# ## NeMo-RL JSONL Export
# - Convert prepared Hugging Face rows into NeMo-RL `ResponseDataset` JSONL records.
# - Preserve prompt, reference answer, reasoning, and response metadata for the reward environment.

# %%
def nemo_jsonl_record(example: dict[str, Any]) -> dict[str, Any]:
    prompt = str(example.get("prompt") or "")
    reference_completion = combine_reasoning_response(
        example.get("reasoning"),
        example.get("response"),
    )
    return {
        "input": prompt,
        "output": reference_completion,
        "messages": [{"role": "user", "content": prompt}],
        "task_name": NEMO_TASK_NAME,
        "final_answer": example.get("final_answer"),
        "reasoning": example.get("reasoning"),
        "response": example.get("response"),
        "source": example.get("source"),
        "answer_type": example.get("answer_type"),
        "dpo_selected": example.get("dpo_selected"),
    }


def write_jsonl_dataset(dataset, output_path: Path, split_name: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in dataset:
            handle.write(json.dumps(nemo_jsonl_record(example), ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset):,} {split_name} examples to {output_path}")
    return output_path


def prepare_nemo_jsonl_datasets(train_dataset, eval_dataset):
    train_jsonl = write_jsonl_dataset(train_dataset, NEMO_TRAIN_JSONL, TRAIN_SPLIT)
    eval_jsonl = None
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_jsonl = write_jsonl_dataset(eval_dataset, NEMO_EVAL_JSONL, EVAL_SPLIT)
    return train_jsonl, eval_jsonl




# %% [markdown]
# ## NeMo-RL Processor and Reward Environment
# - Register a custom `ResponseDataset` processor that carries reward metadata into `extra_env_info`.
# - Implement a Ray environment that scores generated assistant completions with `unified_reward()`.

# %%
def nemotron_grpo_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: Any,
    tokenizer: Any,
    max_seq_length: int,
    idx: int,
) -> dict[str, Any]:
    import torch

    prompt = str(datum_dict.get("input") or "")
    system_prompt = getattr(task_data_spec, "system_prompt", None)
    prompt_template = getattr(task_data_spec, "prompt", None)
    if prompt_template:
        prompt = prompt_template.format(prompt)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": prompt})

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    token_ids = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    loss_multiplier = 1.0
    if token_ids.numel() >= max_seq_length:
        token_ids = token_ids[:max_seq_length]
        loss_multiplier = 0.0

    return {
        "message_log": [
            {
                "role": "user",
                "content": rendered_prompt,
                "token_ids": token_ids,
            }
        ],
        "length": int(token_ids.numel()),
        "extra_env_info": {
            "ground_truth": datum_dict.get("final_answer"),
            "final_answer": datum_dict.get("final_answer"),
            "reasoning": datum_dict.get("reasoning"),
            "response": datum_dict.get("response"),
            "prompt": prompt,
            "source": datum_dict.get("source"),
        },
        "loss_multiplier": torch.tensor(loss_multiplier, dtype=torch.float32),
        "idx": idx,
        "task_name": datum_dict.get("task_name", NEMO_TASK_NAME),
    }


def _assistant_completion_from_log(message_log: list[dict[str, Any]]) -> str:
    assistant_chunks: list[str] = []
    for message in message_log:
        if message.get("role") == "assistant":
            assistant_chunks.append(str(message.get("content") or ""))
    if assistant_chunks:
        return "".join(assistant_chunks)
    if message_log:
        return str(message_log[-1].get("content") or "")
    return ""


def _metadata_list(metadata: Any, batch_size: int) -> list[dict[str, Any]]:
    if metadata is None:
        return [{} for _ in range(batch_size)]
    if isinstance(metadata, list):
        return [item if isinstance(item, dict) else {} for item in metadata]
    if isinstance(metadata, dict):
        return [metadata for _ in range(batch_size)]
    return [{} for _ in range(batch_size)]


def define_nemotron_reward_environment():
    import ray
    import torch
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

    @ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=-1)
    class NemotronUnifiedRewardEnvironment(EnvironmentInterface[dict[str, Any]]):
        def __init__(self, cfg: dict[str, Any] | None = None):
            self.cfg = cfg or {}

        def step(
            self,
            message_log_batch: list[list[dict[str, Any]]],
            metadata: list[dict[str, Any]] | dict[str, Any] | None = None,
        ):
            completions = [
                _assistant_completion_from_log(message_log)
                for message_log in message_log_batch
            ]
            metadata_batch = _metadata_list(metadata, len(completions))
            prompts = [str(item.get("prompt") or "") for item in metadata_batch]
            scores = unified_reward(
                prompts=prompts,
                completions=completions,
                response=[item.get("response") for item in metadata_batch],
                reasoning=[item.get("reasoning") for item in metadata_batch],
                final_answer=[item.get("final_answer") for item in metadata_batch],
            )
            rewards = torch.tensor(scores, dtype=torch.float32)
            answers = [extract_final_answer(completion) for completion in completions]
            return EnvironmentReturn(
                observations=[
                    {
                        "role": "environment",
                        "content": f"Reward: {float(score):.4f}",
                    }
                    for score in scores
                ],
                metadata=metadata_batch,
                next_stop_strings=[None for _ in scores],
                rewards=rewards,
                terminateds=torch.ones(len(scores), dtype=torch.bool),
                answers=answers,
            )

        def global_post_process_and_metrics(self, batch: BatchedDataDict):
            rewards = batch.get("rewards") if hasattr(batch, "get") else None
            if rewards is None:
                return batch, {}
            reward_tensor = rewards.detach().float().cpu()
            metrics = {
                "reward/mean": float(reward_tensor.mean().item()),
                "reward/max": float(reward_tensor.max().item()),
                "reward/min": float(reward_tensor.min().item()),
                "reward/exactish_rate": float((reward_tensor >= EXACT_MATCH_WEIGHT).float().mean().item()),
            }
            return batch, metrics

    globals()["NemotronUnifiedRewardEnvironment"] = NemotronUnifiedRewardEnvironment
    return NemotronUnifiedRewardEnvironment




# %% [markdown]
# ## NeMo-RL Configuration Builder
# - Load a NeMo-RL GRPO YAML config and overlay this project’s model, LoRA, vLLM, and data settings.
# - Enable colocated vLLM generation and sequence-level importance sampling for GSPO-style updates.

# %%
def update_config(config: Any, key: str, value: Any) -> None:
    from omegaconf import OmegaConf

    OmegaConf.update(config, key, value, merge=True, force_add=True)


def resolve_nemo_base_config() -> Path:
    config_path = Path(NEMO_BASE_CONFIG)
    if config_path.exists():
        return config_path
    candidate = Path.cwd() / NEMO_BASE_CONFIG
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Set NEMO_BASE_CONFIG to a valid NeMo-RL GRPO config, for example "
        "examples/configs/grpo_math_1B.yaml from NVIDIA-NeMo/RL."
    )


def build_nemo_config(train_jsonl: Path, eval_jsonl: Path | None):
    from omegaconf import OmegaConf

    config = OmegaConf.load(resolve_nemo_base_config())
    adapter_input_path = prepare_adapter_input_path()
    model_name = adapter_input_path or MODEL_PATH

    update_config(config, "grpo.num_generations_per_prompt", NUM_GENERATIONS)
    update_config(config, "grpo.max_num_steps", MAX_STEPS if MAX_STEPS > 0 else -1)
    update_config(config, "grpo.max_num_epochs", NUM_TRAIN_EPOCHS)
    update_config(config, "grpo.val_period", EVAL_STEPS)
    update_config(config, "grpo.val_at_start", eval_jsonl is not None)
    update_config(config, "grpo.val_at_end", eval_jsonl is not None)
    update_config(config, "grpo.normalize_rewards", NEMO_NORMALIZE_REWARDS)
    update_config(config, "loss_fn.sequence_level_importance_ratios", NEMO_SEQUENCE_LEVEL_IS)
    update_config(config, "policy.model_name", str(model_name))
    update_config(config, "policy.tokenizer.name", str(MODEL_PATH))
    update_config(config, "policy.train_global_batch_size", NEMO_TRAIN_GLOBAL_BATCH_SIZE)
    update_config(config, "policy.train_micro_batch_size", PER_DEVICE_TRAIN_BATCH_SIZE)
    update_config(config, "policy.generation_batch_size", NEMO_GENERATION_BATCH_SIZE)
    update_config(config, "policy.max_total_sequence_length", MAX_SEQ_LENGTH)
    update_config(config, "policy.max_grad_norm", 0.1)
    update_config(config, "policy.precision", "bfloat16")
    update_config(config, "policy.optimizer.name", "torch.optim.AdamW")
    update_config(config, "policy.optimizer.kwargs.lr", LEARNING_RATE)
    update_config(config, "policy.optimizer.kwargs.betas", [0.9, 0.99])
    update_config(config, "policy.optimizer.kwargs.eps", 1e-8)
    update_config(config, "policy.optimizer.kwargs.weight_decay", 0.0)
    update_config(config, "policy.lora_cfg.enabled", True)
    update_config(config, "policy.lora_cfg.dim", LORA_R)
    update_config(config, "policy.lora_cfg.alpha", LORA_ALPHA)
    update_config(config, "policy.lora_cfg.dropout", 0.0)
    update_config(
        config,
        "policy.lora_cfg.target_modules",
        [
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
    )
    update_config(config, "policy.generation.backend", "vllm")
    update_config(config, "policy.generation.temperature", TEMPERATURE)
    update_config(config, "policy.generation.top_p", TOP_P)
    update_config(config, "policy.generation.top_k", TOP_K)
    update_config(config, "policy.generation.max_new_tokens", MAX_COMPLETION_LENGTH)
    update_config(config, "policy.generation.vllm_cfg.gpu_memory_utilization", VLLM_GPU_MEMORY_UTILIZATION)
    update_config(config, "policy.generation.vllm_cfg.tensor_parallel_size", NEMO_TENSOR_PARALLEL_SIZE)
    update_config(config, "policy.generation.vllm_cfg.pipeline_parallel_size", NEMO_PIPELINE_PARALLEL_SIZE)
    update_config(config, "policy.generation.colocated.enabled", True)
    update_config(config, "data.train.data_path", str(train_jsonl))
    update_config(config, "data.train.input_key", "input")
    update_config(config, "data.train.output_key", "output")
    update_config(config, "data.train.dataset_name", "ResponseDataset")
    update_config(config, "data.train.env_name", NEMO_ENV_NAME)
    if eval_jsonl is not None:
        update_config(config, "data.validation.data_path", str(eval_jsonl))
        update_config(config, "data.validation.input_key", "input")
        update_config(config, "data.validation.output_key", "output")
        update_config(config, "data.validation.dataset_name", "ResponseDataset")
        update_config(config, "data.validation.env_name", NEMO_ENV_NAME)
    else:
        update_config(config, "data.validation", None)
    update_config(config, "data.default.dataset_name", "ResponseDataset")
    update_config(config, "data.default.input_key", "input")
    update_config(config, "data.default.output_key", "output")
    update_config(config, "data.default.processor", NEMO_PROCESSOR_NAME)
    update_config(config, "data.default.env_name", NEMO_ENV_NAME)
    update_config(config, "data.max_input_seq_length", MAX_PROMPT_LENGTH)
    update_config(config, f"env.{NEMO_ENV_NAME}.num_workers", NEMO_ENV_NUM_WORKERS)
    update_config(config, "logger.log_dir", str(OUTPUT_DIR))
    update_config(config, "logger.wandb_enabled", "wandb" in REPORT_TO)
    update_config(config, "logger.wandb.name", RUN_NAME)
    update_config(config, "checkpointing.checkpoint_dir", str(ADAPTER_OUTPUT_DIR))
    update_config(config, "checkpointing.save_period", SAVE_STEPS)
    update_config(config, "checkpointing.save_at_end", True)
    return config




# %% [markdown]
# ## Training Runtime Bootstrap
# - Materialize train and validation splits, then export them to NeMo-RL JSONL files.
# - Run a local reward sanity check before invoking NeMo-RL.

# %%
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

train_dataset, eval_dataset = prepare_datasets()
train_jsonl, eval_jsonl = prepare_nemo_jsonl_datasets(train_dataset, eval_dataset)

print("NeMo-RL GRPO/GSPO training sample:")
print(train_dataset[0])
sample_reward = unified_reward(
    prompts=[train_dataset[0]["prompt"]],
    completions=[
        (
            "<think>\nExample reasoning\n</think>\n"
            f"\\boxed{{{train_dataset[0]['final_answer']}}}"
        )
    ],
    response=[train_dataset[0]["response"]],
    reasoning=[train_dataset[0]["reasoning"]],
    final_answer=[train_dataset[0]["final_answer"]],
)
print(f"Reward sanity check: {sample_reward}")




# %% [markdown]
# ## NeMo-RL Training Execution
# - Register the custom processor and reward environment with NeMo-RL.
# - Start single-node GRPO training with colocated vLLM generation when `NEMO_RUN_TRAIN=1`.

# %%
def select_nemo_grpo_trainer(master_config):
    data_plane_config = master_config.data_plane or {}
    if data_plane_config.get("enabled", False):
        from nemo_rl.algorithms.grpo_sync import grpo_train_sync

        print("Running synchronous NeMo-RL GRPO training with TransferQueue")
        return grpo_train_sync
    from nemo_rl.algorithms.grpo import grpo_train

    print("Running synchronous NeMo-RL GRPO training with legacy data path")
    return grpo_train


def run_nemo_grpo_training(train_jsonl: Path, eval_jsonl: Path | None):
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig, setup
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.processors import register_processor
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.environments.utils import register_env
    from nemo_rl.models.generation import configure_generation_config

    define_nemotron_reward_environment()
    register_processor(NEMO_PROCESSOR_NAME, nemotron_grpo_data_processor)
    register_env(NEMO_ENV_NAME, NEMO_ENV_FQN)

    nemo_config = build_nemo_config(train_jsonl, eval_jsonl)
    config_dict = OmegaConf.to_container(nemo_config, resolve=True)
    master_config = MasterConfig(**config_dict)
    init_ray()

    tokenizer = get_tokenizer(master_config.policy["tokenizer"])
    has_refit_draft_weights = bool(
        master_config.policy.get("draft", {}).get("enabled", False)
    )
    master_config.policy["generation"] = configure_generation_config(
        master_config.policy.get("generation", {}),
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
    )
    train_data, val_data, task_to_env, val_task_to_env = setup_response_data(
        tokenizer,
        master_config.data,
        master_config.env,
    )
    data_plane_config = master_config.data_plane or {}
    if data_plane_config.get("enabled", False):
        from nemo_rl.models.policy.tq_policy import TQPolicy

        def policy_factory(**kwargs):
            return TQPolicy(**kwargs, dp_cfg=data_plane_config)
    else:
        policy_factory = None

    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(
        master_config,
        tokenizer,
        train_data,
        val_data,
        policy_factory=policy_factory,
    )
    trainer = select_nemo_grpo_trainer(master_config)
    trainer(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )
    return master_config


nemo_master_config = None
if NEMO_RUN_TRAIN:
    nemo_master_config = run_nemo_grpo_training(train_jsonl, eval_jsonl)
else:
    print("Skipping NeMo-RL training because NEMO_RUN_TRAIN=0")




# %% [markdown]
# ## Adapter Metadata Normalization
# - Rewrite saved adapter metadata back from Kaggle-local model paths to the original base model id.
# - Keep uploaded adapter configs portable outside the Kaggle runtime.

# %%
if MODEL_PATH and BASE_MODEL_ID:
    for metadata_path in [
        ADAPTER_OUTPUT_DIR / "README.md",
        ADAPTER_OUTPUT_DIR / "adapter_config.json",
    ]:
        if metadata_path.exists():
            metadata_path.write_text(
                metadata_path.read_text().replace(MODEL_PATH, BASE_MODEL_ID)
            )

print(f"NeMo-RL checkpoints/adapters are under {ADAPTER_OUTPUT_DIR}")




# %% [markdown]
# ## Upload Artifacts
# - Publish NeMo-RL checkpoint or adapter folders when upload flags are enabled.
# - Use folder uploads because NeMo-RL owns checkpoint serialization.

# %%
if PUSH_TO_HUB:
    try:
        if not HF_KEY:
            raise RuntimeError("PUSH_TO_HUB=1 but HF_KEY/HF_TOKEN is not configured")
        from huggingface_hub import HfApi, login

        login(token=HF_KEY)
        api = HfApi()
        api.create_repo(
            repo_id=HF_ADAPTER_REPO,
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=ADAPTER_OUTPUT_DIR,
            repo_id=HF_ADAPTER_REPO,
            repo_type="model",
        )
        print("Upload to Hugging Face succeeded")
    except Exception as exc:
        print(f"Upload to Hugging Face failed: {exc}")




# %% [markdown]
# ## Upload Artifacts
# - Publish adapter/checkpoint folders to Kaggle when requested.
# - Remove transient local Kaggle state after successful model upload.

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
            version_notes=f"Nemotron LoRA continued with NeMo-RL {TRAIN_STAGE}",
            license_name="Apache 2.0",
        )
        (WORKING_DIR / "state.db").unlink(missing_ok=True)
        print("Upload to Kaggle succeeded")
    except Exception as exc:
        print(f"Upload to Kaggle failed: {exc}")
