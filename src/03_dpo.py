# %% [markdown]
# ## DPO Training Overview
# - Continues the SFT LoRA adapter with direct preference optimization.
# - Filters rows with non-empty chosen/rejected fields and formats prompt/completion pairs
#   for DPO.
# - Uses Unsloth model loading and TRL `DPOTrainer` with configurable beta, loss type, and
#   reference log-prob handling.
# - Saves the continued adapter and optional Hugging Face/Kaggle artifacts.

# %% [markdown]
# ## Imports
# - Load dependencies for this notebook script.
# - Set early runtime flags before heavier stage-specific imports run.

# %%
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional



# %% [markdown]
# ## Credentials
# - Read Kaggle, Hugging Face, and W&B credentials when available.
# - Keep committed defaults safe for public or local dry runs.

# %%
# User secrets
# try:
#     from kaggle_secrets import UserSecretsClient  # type: ignore

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

# if KAGGLE_KEY:
#     os.environ["KAGGLE_KEY"] = KAGGLE_KEY
# if KAGGLE_USERNAME:
#     os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
# if HF_KEY:
#     os.environ["HF_TOKEN"] = HF_KEY
# if WANDB_KEY:
#     os.environ["WANDB_API_KEY"] = WANDB_KEY

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
#     "transformers>=4.56.2,<5.0.0" \
#     "tokenizers>=0.22.0,<=0.23.0" \
#     "trl>=0.22.2" \
#     unsloth \
#     unsloth_zoo \
#     --no-index --find-links={wheels_dir}
# !uv pip install --no-deps "torchcodec==0.10.0+cu128" --no-index --find-links={wheels_dir}
# !uv pip install mamba_ssm causal_conv1d --no-index --find-links={wheels_dir}
# !uv pip install --no-deps --upgrade "torchao>=0.16.0" \
#     --no-index --find-links={wheels_dir}
# # !uv pip install "vllm>=0.12.0,<0.19.0" --no-index --find-links={wheels_dir}
# # !uv pip install "protobuf<6.0.0" --no-index --find-links={wheels_dir}



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
    "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/lora-sft/7",
)
BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/kaggle/input/datasets/rohitraje0493/nemotron-reasoning",
)
DATASET_REVISION = os.environ.get("DATASET_REVISION")
HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", "/tmp/hf_cache"))
TRAIN_SPLIT = os.environ.get("TRAIN_SPLIT", "train")
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "validation")

def optional_nonnegative_int(name: str, default: Optional[int] = None) -> Optional[int]:
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
TRAIN_MAX_IDX = optional_nonnegative_int("TRAIN_MAX_IDX")
EVAL_MIN_IDX = optional_nonnegative_int("EVAL_MIN_IDX")
EVAL_MAX_IDX = optional_nonnegative_int("EVAL_MAX_IDX")
SOURCE_OPTIONS = {
    TRAIN_SPLIT: {
        "include": optional_string_list("TRAIN_INCLUDE_SOURCES", '["nvidia-nemotron-model-reasoning-challenge", "dgxchen/nemotron-cot-tong", "BytedTsinghua-SIA/Enigmata-Eval"]'),
        "order": optional_string_list("TRAIN_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("TRAIN_EXCLUDE_SOURCES"),
        "order_remaining": os.environ.get(
            "TRAIN_ORDER_REMAINING",
            "0",
        ).lower() not in {"0", "false", "no"},
    },
    EVAL_SPLIT: {
        "include": optional_string_list("EVAL_INCLUDE_SOURCES", '["BytedTsinghua-SIA/Enigmata-Eval"]'),
        "order": optional_string_list("EVAL_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("EVAL_EXCLUDE_SOURCES"),
        "order_remaining": os.environ.get(
            "EVAL_ORDER_REMAINING",
            "0",
        ).lower() not in {"0", "false", "no"},
    },
}
TRAIN_SHUFFLE = os.environ.get("TRAIN_SHUFFLE", "1").lower() not in {
    "0",
    "false",
    "no",
}
EVAL_SHUFFLE = os.environ.get("EVAL_SHUFFLE", "0").lower() not in {
    "0",
    "false",
    "no",
}
FILTER_HQ_BY_SPLIT = {
    TRAIN_SPLIT: os.environ.get("TRAIN_FILTER_HQ", "0").lower()
        not in {"0", "false", "no"},
    EVAL_SPLIT: os.environ.get("EVAL_FILTER_HQ", "0").lower()
        not in {"0", "false", "no"},
}

TRAIN_STAGE = os.environ.get("TRAIN_STAGE", "dpo")
TRAIN_VERSION = os.environ.get("TRAIN_VERSION", "v6")
RUN_NAME = os.environ.get("RUN_NAME", f"nemotron-{TRAIN_STAGE}-{TRAIN_VERSION}")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKING_DIR / RUN_NAME)))
ADAPTER_OUTPUT_DIR = Path(
    os.environ.get(
        "ADAPTER_OUTPUT_DIR",
        str(WORKING_DIR / f"nemotron-lora-{TRAIN_STAGE}-{TRAIN_VERSION}"),
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
MAX_COMPLETION_LENGTH = optional_nonnegative_int("MAX_COMPLETION_LENGTH")
# if MAX_COMPLETION_LENGTH is None:
#     MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH
# if MAX_PROMPT_LENGTH <= 0 or MAX_COMPLETION_LENGTH <= 0:
#     raise ValueError("Prompt and completion lengths must be positive")
# if MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH > MAX_SEQ_LENGTH:
#     raise ValueError(
#         "MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH must not exceed "
#         "MAX_SEQ_LENGTH"
#     )

DATASET_WORKERS = max(1, int(os.environ.get("DATASET_NUM_PROC", "8")))
DATASET_NUM_PROC = DATASET_WORKERS if DATASET_WORKERS > 1 else None
SEED = int(os.environ.get("SEED", "3407"))

PER_DEVICE_TRAIN_BATCH_SIZE = int(
    os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1")
)
PER_DEVICE_EVAL_BATCH_SIZE = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")
)
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "2"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-6"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "1"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "5"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(SAVE_STEPS)))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))
LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))

DPO_BETA = float(os.environ.get("DPO_BETA", "0.1"))
DPO_LOSS_TYPE = os.environ.get("DPO_LOSS_TYPE", "sigmoid")
PRECOMPUTE_REF_LOG_PROBS = os.environ.get(
    "PRECOMPUTE_REF_LOG_PROBS",
    "1",
).lower() in {"1", "true", "yes"}
REPORT_TO = os.environ.get("REPORT_TO", "wandb")
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "0")
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "0").lower() in {
    "1",
    "true",
    "yes",
}
PUSH_TO_KAGGLE = os.environ.get("PUSH_TO_KAGGLE", "0").lower() in {
    "1",
    "true",
    "yes",
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

HAS_DIGIT_RE = re.compile(r"\d")




# %% [markdown]
# ## Helper Functions
# - Define reusable helpers including `clean_text`, `is_preference_example`,
#   `is_high_quality_example`, `build_user_content`, `remove_leading_think`,
#   `render_dpo_example`.
# - Support parsing, filtering, formatting, verification, loading, or upload behavior used
#   later.

# %%
def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def is_preference_example(example: dict[str, Any]) -> bool:
    return (
        clean_text(example.get("prompt")) is not None
        and clean_text(example.get("chosen")) is not None
        and clean_text(example.get("rejected")) is not None
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
    if (example.get("source").casefold() == "BytedTsinghua-SIA/Enigmata-Eval".casefold() 
        and HAS_DIGIT_RE.search(example.get("final_answer"))):
        return True
    answer_type = clean_text(example.get("answer_type"))
    if answer_type is not None and answer_type.lower() in HQ_ANSWER_TYPES:
        return True
    return False
    # final_answer = clean_text(example.get("final_answer"))
    # return final_answer is not None and final_answer.isalnum()


def build_user_content(prompt: Any) -> str:
    normalized_prompt = clean_text(prompt)
    if normalized_prompt is None:
        raise ValueError("Cannot format DPO data without a prompt")
    return normalized_prompt + BOXED_ANSWER_INSTRUCTION


THINK_OPEN_RE = re.compile(r"^\s*<think>\s*", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"\s*</think>\s*", flags=re.IGNORECASE)


def remove_leading_think(value: str, remove_closing: bool = False) -> str:
    has_opening = THINK_OPEN_RE.match(value) is not None
    has_closing = remove_closing and THINK_CLOSE_RE.search(value) is not None
    if not has_opening and not has_closing:
        return value
    value = THINK_OPEN_RE.sub("", value, count=1)
    if remove_closing:
        value = THINK_CLOSE_RE.sub("\n", value, count=1)
    return value.lstrip()


def render_dpo_example(
    example: dict[str, Any],
    tokenizer: Any,
) -> dict[str, str]:
    chosen = clean_text(example.get("chosen"))
    rejected = clean_text(example.get("rejected"))
    if chosen is None or rejected is None:
        raise ValueError("Dataset must be filtered before DPO formatting")

    prompt_messages = [
        {
            "role": "user",
            "content": build_user_content(example.get("prompt")),
        }
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    stripped_prompt = prompt.rstrip()
    if stripped_prompt.endswith("<think></think>"):
        chosen = remove_leading_think(chosen, remove_closing=True)
        rejected = remove_leading_think(rejected, remove_closing=True)
    elif stripped_prompt.endswith("<think>"):
        chosen = remove_leading_think(chosen, remove_closing=False)
        rejected = remove_leading_think(rejected, remove_closing=False)
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }




# %% [markdown]
# ## Dataset Loading
# - Load datasets from local disk, Kaggle-mounted paths, Hugging Face repos, parquet files,
#   or saved DatasetDicts.
# - Normalize split handling and cache behavior for downstream processing.

# %%
def load_preference_dataset():
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
    if not include_sources and not order_sources and not exclude_sources:
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
    before_remaining_indices = [
        index
        for source in before_remaining_sources
        for index in indices_by_source.get(source, [])
    ]
    if order_remaining:
        remaining_indices = [
            index
            for source in remaining_sources
            for index in indices_by_source.get(source, [])
        ]
    else:
        remaining_source_set = set(remaining_sources)
        remaining_indices = [
            index
            for index, source in enumerate(dataset["source"])
            if source in remaining_source_set
        ]
    after_remaining_indices = [
        index
        for source in after_remaining_sources
        for index in indices_by_source.get(source, [])
    ]
    selected_indices = (
        before_remaining_indices
        + remaining_indices
        + after_remaining_indices
    )
    ordered = dataset.select(selected_indices, keep_in_memory=KEEP_IN_MEMORY)
    print(
        f"{split_name}: source order={ordered_sources}; "
        f"order_remaining={order_remaining}; retained={len(ordered):,}"
    )
    return ordered


def prepare_split(
    dataset,
    tokenizer,
    split_name: str,
    allow_empty: bool = False,
):
    original_size = len(dataset)
    dataset = dataset.filter(
        is_preference_example,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: keep complete preferences",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    if FILTER_HQ_BY_SPLIT.get(split_name):
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
            print(f"{split_name}: no preference examples after filtering")
            return None
        raise ValueError(
            f"{split_name} has no preference examples after filtering"
        )

    dataset = dataset.map(
        render_dpo_example,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: apply DPO chat template",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    if dataset.column_names != ["prompt", "chosen", "rejected"]:
        raise RuntimeError(
            f"{split_name}: unexpected DPO columns {dataset.column_names}"
        )
    print(f"{split_name}: retained {len(dataset):,}/{original_size:,} examples")
    return dataset


def select_index_range(
    dataset,
    min_idx: Optional[int],
    max_idx: Optional[int],
    split_name: str,
    shuffle: bool = False,
    seed: int = SEED,
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
        else dataset.select(range(start, stop), keep_in_memory=KEEP_IN_MEMORY)
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
        selected = selected.shuffle(seed=seed, keep_in_memory=KEEP_IN_MEMORY)
        print(f"{split_name}: shuffled {len(selected):,} examples with seed {seed}")
    return selected


def prepare_datasets(tokenizer):
    dataset_dict = load_preference_dataset()
    train_source_dataset = apply_source_options(
        dataset_dict[TRAIN_SPLIT],
        TRAIN_SPLIT,
    )
    train_dataset = prepare_split(
        train_source_dataset,
        tokenizer,
        TRAIN_SPLIT,
    )
    train_dataset = select_index_range(
        train_dataset,
        TRAIN_MIN_IDX,
        TRAIN_MAX_IDX,
        TRAIN_SPLIT,
        TRAIN_SHUFFLE,
        SEED,
    )

    eval_dataset = None
    if EVAL_SPLIT in dataset_dict and len(dataset_dict[EVAL_SPLIT]):
        eval_source_dataset = apply_source_options(
            dataset_dict[EVAL_SPLIT],
            EVAL_SPLIT,
        )
        eval_dataset = prepare_split(
            eval_source_dataset,
            tokenizer,
            EVAL_SPLIT,
            allow_empty=True,
        )
        if eval_dataset is not None:
            eval_dataset = select_index_range(
                eval_dataset,
                EVAL_MIN_IDX,
                EVAL_MAX_IDX,
                EVAL_SPLIT,
                EVAL_SHUFFLE,
                SEED,
            )
    return train_dataset, eval_dataset




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
    from unsloth import FastLanguageModel

    adapter_input_path = prepare_adapter_input_path()
    model_source = adapter_input_path or MODEL_PATH

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        token=HF_KEY,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization=0.95,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if adapter_input_path is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
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
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
            use_rslora=False,
            loftq_config=None,
        )
    else:
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if trainable_parameters == 0:
            raise RuntimeError(
                "The SFT adapter loaded without trainable parameters; "
                "DPO would not update the existing LoRA"
            )
        print(f"Trainable adapter parameters: {trainable_parameters:,}")
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
# - Initialize tokenizer and CUDA allocator settings.
# - Create output directories and import Unsloth DPO patching, Torch, and TRL DPO utilities.
# %%
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from unsloth import FastLanguageModel, PatchDPOTrainer  # noqa: F401

PatchDPOTrainer()

import torch
from trl import DPOConfig, DPOTrainer

if not torch.cuda.is_available():
    raise RuntimeError("DPO training requires a CUDA GPU")



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
train_dataset, eval_dataset = prepare_datasets(tokenizer)



# %% [markdown]
# ## Trainer Configuration
# - Configure trainer-specific hyperparameters, precision, checkpointing, evaluation, and
#   reporting.
# - Bind optimizer, batching, sequence length, and stage-specific training options.

# %%
has_eval = eval_dataset is not None and len(eval_dataset) > 0
bf16 = torch.cuda.is_bf16_supported()

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=DPOConfig(
        output_dir=str(OUTPUT_DIR),
        run_name=RUN_NAME,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        dataset_num_proc=DATASET_WORKERS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=SEED,
        data_seed=SEED,
        report_to=REPORT_TO,
        bf16=bf16,
        fp16=not bf16,
        tf32=None,
        padding_free=False,
        beta=DPO_BETA,
        loss_type=[DPO_LOSS_TYPE],
        precompute_ref_log_probs=PRECOMPUTE_REF_LOG_PROBS,
        max_grad_norm=0.3,
    ),
)



# %% [markdown]
# ## Runtime Sanity Checks
# - Print representative samples, reward values, GPU details, or runtime metrics.
# - Help catch formatting and resource issues before or after long-running work.

# %%
sample = train_dataset[0]
print("Rendered DPO prompt:")
print(sample["prompt"][:2000])
print("Chosen completion:")
print(sample["chosen"][:2000])
print("Rejected completion:")
print(sample["rejected"][:2000])

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
trainer_stats = dpo_trainer.train(
    resume_from_checkpoint=resolve_resume_from_checkpoint()
)



# %% [markdown]
# ## Save Artifacts
# - Persist datasets, adapters, tokenizer files, trainer state, or output folders.
# - Prepare generated artifacts for reuse and optional publishing.

# %%
dpo_trainer.save_state()
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
print(f"Saved DPO-trained LoRA adapter to {ADAPTER_OUTPUT_DIR}")



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
