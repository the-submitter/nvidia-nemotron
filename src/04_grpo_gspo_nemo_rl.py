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
import os
import shutil
import subprocess
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

def bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}

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

DATASET_WORKERS = max(1, int(os.environ.get("DATASET_NUM_PROC", "8")))
DATASET_NUM_PROC = DATASET_WORKERS if DATASET_WORKERS > 1 else None
SEED = int(os.environ.get("SEED", "3407"))

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
from nemo_bridge.nemotron_reward_utils import (
    clean_text,
    combine_reasoning_response,
    unified_reward,
)

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




# %% [markdown]
# ## NeMo-RL JSONL Settings
# - Configure response-dataset JSONL locations consumed by NeMo-RL.
# - Keep backend-specific knobs isolated from the common dataset and reward code.

# %%
NEMO_DATA_DIR = Path(os.environ.get("NEMO_DATA_DIR", WORKING_DIR / "nemo_rl_data"))
NEMO_TRAIN_JSONL = NEMO_DATA_DIR / "train.jsonl"
NEMO_EVAL_JSONL = NEMO_DATA_DIR / "validation.jsonl"
NEMO_RL_DIR = Path(os.environ.get("NEMO_RL_DIR", str(WORKING_DIR / "nemo-rl")))
NEMO_CONFIG_PATH = Path(
    os.environ.get("NEMO_CONFIG_PATH", "configs/grpo_gspo_nemotron.yaml")
)
NEMO_BRIDGE_DIR = Path(globals().get("__file__", Path.cwd() / "src" / "04_grpo_gspo_nemo_rl.py")).resolve().parent / "nemo_bridge"
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
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reference_completion},
        ],
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
# ## NeMo-RL Config and Bridge
# - Resolve the static NeMo-RL YAML config used by `examples/run_grpo.py --config`.
# - Validate that the exported JSONL paths match the static config before launch.

# %%
def resolve_nemo_config_path() -> Path:
    if NEMO_CONFIG_PATH.exists():
        return NEMO_CONFIG_PATH.resolve()
    candidate = (Path.cwd() / NEMO_CONFIG_PATH).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"NeMo config not found: {NEMO_CONFIG_PATH}")


def validate_nemo_config_paths(config_path: Path, train_jsonl: Path, eval_jsonl: Path | None) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    configured_train = Path(str(config.data.train.data_path)).resolve()
    actual_train = train_jsonl.resolve()
    if configured_train != actual_train:
        raise ValueError(
            "NeMo config train data_path does not match exported JSONL: "
            f"{configured_train} != {actual_train}. Update {config_path}."
        )
    if config.data.validation is None:
        if eval_jsonl is not None:
            print("Validation JSONL was exported, but NeMo config validation is null")
        return
    if eval_jsonl is None:
        raise ValueError(
            f"NeMo config expects validation data at {config.data.validation.data_path}, "
            "but no eval dataset was exported. Set validation: null or configure EVAL_SPLIT."
        )
    configured_eval = Path(str(config.data.validation.data_path)).resolve()
    actual_eval = eval_jsonl.resolve()
    if configured_eval != actual_eval:
        raise ValueError(
            "NeMo config validation data_path does not match exported JSONL: "
            f"{configured_eval} != {actual_eval}. Update {config_path}."
        )


def validate_nemo_bridge() -> None:
    bridge_module = NEMO_BRIDGE_DIR / "nemotron_nemo_bridge.py"
    sitecustomize_module = NEMO_BRIDGE_DIR / "sitecustomize.py"
    if not bridge_module.exists() or not sitecustomize_module.exists():
        raise FileNotFoundError(
            f"Expected NeMo bridge files under {NEMO_BRIDGE_DIR}"
        )




# %% [markdown]
# ## Training Runtime Bootstrap
# - Materialize train and validation splits, then export them to NeMo-RL JSONL files.
# - Check that the static NeMo config and bridge module are ready for subprocess launch.

# %%
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

train_dataset, eval_dataset = prepare_datasets()
train_jsonl, eval_jsonl = prepare_nemo_jsonl_datasets(train_dataset, eval_dataset)
nemo_config_path = resolve_nemo_config_path()
validate_nemo_config_paths(nemo_config_path, train_jsonl, eval_jsonl)
validate_nemo_bridge()

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
# ## NeMo-RL Subprocess Launch
# - Run `uv run python examples/run_grpo.py --config <nemotron-yaml>` from `NEMO_RL_DIR`.
# - Put `src/nemo_bridge` on `PYTHONPATH` so `sitecustomize` registers the processor and environment.

# %%
def nemo_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(NEMO_BRIDGE_DIR)
        if not existing_pythonpath
        else f"{NEMO_BRIDGE_DIR}{os.pathsep}{existing_pythonpath}"
    )
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("TRANSFORMERS_NO_FLAX", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    return env


def run_nemo_grpo_subprocess(config_path: Path) -> subprocess.CompletedProcess[str] | None:
    if not NEMO_RUN_TRAIN:
        print("Skipping NeMo-RL training because NEMO_RUN_TRAIN=0")
        return None
    if not NEMO_RL_DIR.exists():
        raise FileNotFoundError(f"NEMO_RL_DIR does not exist: {NEMO_RL_DIR}")
    if not (NEMO_RL_DIR / "examples" / "run_grpo.py").exists():
        raise FileNotFoundError(f"examples/run_grpo.py not found under {NEMO_RL_DIR}")
    command = ["uv", "run", "python", "examples/run_grpo.py", "--config", str(config_path)]
    print(f"Running NeMo-RL subprocess in {NEMO_RL_DIR}: {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=str(NEMO_RL_DIR),
        env=nemo_subprocess_env(),
        check=True,
        text=True,
    )


nemo_subprocess_result = run_nemo_grpo_subprocess(nemo_config_path)




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
