# %%
from __future__ import annotations

import csv
import json
import math
import multiprocessing
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# %%
# User secrets
# try:
#     from kaggle_secrets import UserSecretsClient  # type: ignore

#     user_secrets = UserSecretsClient()
#     KAGGLE_KEY = user_secrets.get_secret("KAGGLE_KEY")
#     KAGGLE_USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
#     HF_KEY = user_secrets.get_secret("HF_KEY")
# except Exception:
#     KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
#     KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
#     HF_KEY = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")

# if KAGGLE_KEY:
#     os.environ["KAGGLE_KEY"] = KAGGLE_KEY
# if KAGGLE_USERNAME:
#     os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
# if HF_KEY:
#     os.environ["HF_TOKEN"] = HF_KEY

HF_KEY = KAGGLE_KEY = KAGGLE_USERNAME = None

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
# !uv pip install "vllm>=0.12.0,<0.19.0" --no-index --find-links={wheels_dir}
# !uv pip install "protobuf<6.0.0" --no-index --find-links={wheels_dir}

# %%
WORKING_DIR = Path(os.environ.get("WORKING_DIR", "/kaggle/working"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    # "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/default/1",
)
LORA_PATH = os.environ.get(
    "LORA_PATH",
    "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/lora-sft/7",
)
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/kaggle/input/datasets/rohitraje0493/nemotron-reasoning",
)
DATASET_REVISION = os.environ.get("DATASET_REVISION")
SPLIT_NAMES = ("train", "validation", "test")
HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", "/tmp/hf_cache"))

DATASET_TAG = os.environ.get("DATASET_TAG", "nemotron-reasoning")
LOCAL_OUTPUT_DIR = Path(
    os.environ.get("LOCAL_OUTPUT_DIR", str(WORKING_DIR / DATASET_TAG))
)
BACKUP_DIR = Path(
    os.environ.get("BACKUP_DIR", str(WORKING_DIR / f"{DATASET_TAG}-backups"))
)
INCREMENTAL_DIR = Path(
    os.environ.get(
        "INCREMENTAL_DIR",
        str(WORKING_DIR / f"{DATASET_TAG}-incremental"),
    )
)
KAGGLE_OUTPUT_DIR = Path(
    os.environ.get(
        "KAGGLE_OUTPUT_DIR",
        str(WORKING_DIR / f"{DATASET_TAG}-kaggle"),
    )
)
KAGGLE_DATASET_REPO = os.environ.get(
    "KAGGLE_DATASET_REPO",
    f"{KAGGLE_USERNAME}/{DATASET_TAG}",
)
HF_UPLOAD_USERNAME = os.environ.get(
    "HF_UPLOAD_USERNAME",
    "the-submitter",
)

TRAJECTORIES = max(1, int(os.environ.get("TRAJECTORIES", "4")))
PROMPT_BATCH_SIZE = max(1, int(os.environ.get("PROMPT_BATCH_SIZE", "100")))
SEED = int(os.environ.get("SEED", "3407"))

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

DEFAULT_TAKE = {
    "missing": {
        "train": 2_000,
        "validation": 50,
        "test": 0,
    },
    "existing": {
        "train": 114,
        "validation": 5,
        "test": 0,
    },
}
CANDIDATE_SELECTION_OPTIONS = {
    "train": {
        "missing": {
            "take": optional_nonnegative_int(
                "TRAIN_MISSING_TAKE",
            ),
            "min_idx": optional_nonnegative_int(
                "TRAIN_MISSING_MIN_IDX",
            ),
            "max_idx": optional_nonnegative_int(
                "TRAIN_MISSING_MAX_IDX",
                DEFAULT_TAKE["missing"]["train"],
            ),
            "filter_hq": os.environ.get(
                "TRAIN_MISSING_FILTER_HQ",
                "1",
            ).lower() not in {"0", "false", "no"},
        },
        "existing": {
            "take": optional_nonnegative_int(
                "TRAIN_EXISTING_TAKE",
            ),
            "min_idx": optional_nonnegative_int(
                "TRAIN_EXISTING_MIN_IDX",
            ),
            "max_idx": optional_nonnegative_int(
                "TRAIN_EXISTING_MAX_IDX",
                DEFAULT_TAKE["existing"]["train"],
            ),
            "filter_hq": os.environ.get(
                "TRAIN_EXISTING_FILTER_HQ",
                "1",
            ).lower() not in {"0", "false", "no"},
        },
    },
    "validation": {
        "missing": {
            "take": optional_nonnegative_int(
                "VALIDATION_MISSING_TAKE",
            ),
            "min_idx": optional_nonnegative_int(
                "VALIDATION_MISSING_MIN_IDX",
            ),
            "max_idx": optional_nonnegative_int(
                "VALIDATION_MISSING_MAX_IDX",
                DEFAULT_TAKE["missing"]["validation"],
            ),
            "filter_hq": os.environ.get(
                "VALIDATION_MISSING_FILTER_HQ",
                "1",
            ).lower() not in {"0", "false", "no"},
        },
        "existing": {
            "take": optional_nonnegative_int(
                "VALIDATION_EXISTING_TAKE",
            ),
            "min_idx": optional_nonnegative_int(
                "VALIDATION_EXISTING_MIN_IDX",
            ),
            "max_idx": optional_nonnegative_int(
                "VALIDATION_EXISTING_MAX_IDX",
                DEFAULT_TAKE["existing"]["validation"],
            ),
            "filter_hq": os.environ.get(
                "VALIDATION_EXISTING_FILTER_HQ",
                "1",
            ).lower() not in {"0", "false", "no"},
        },
    },
    "test": {
        "missing": {
            "take": optional_nonnegative_int(
                "TEST_MISSING_TAKE",
            ),
            "min_idx": optional_nonnegative_int(
                "TEST_MISSING_MIN_IDX",
            ),
            "max_idx": optional_nonnegative_int(
                "TEST_MISSING_MAX_IDX",
                DEFAULT_TAKE["missing"]["test"],
            ),
            "filter_hq": os.environ.get(
                "TEST_MISSING_FILTER_HQ",
                "1",
            ).lower() not in {"0", "false", "no"},
        },
        "existing": {
            "take": optional_nonnegative_int(
                "TEST_EXISTING_TAKE",
            ),
            "min_idx": optional_nonnegative_int(
                "TEST_EXISTING_MIN_IDX",
            ),
            "max_idx": optional_nonnegative_int(
                "TEST_EXISTING_MAX_IDX",
                DEFAULT_TAKE["existing"]["test"],
            ),
            "filter_hq": os.environ.get(
                "TEST_EXISTING_FILTER_HQ",
                "1",
            ).lower() not in {"0", "false", "no"},
        },
    },
}
SOURCE_OPTIONS = {
    "train": {
        "include": optional_string_list("TRAIN_INCLUDE_SOURCES", '["nvidia-nemotron-model-reasoning-challenge", "dgxchen/nemotron-cot-tong"]'),
        "order": optional_string_list("TRAIN_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("TRAIN_EXCLUDE_SOURCES"),
        "order_remaining": os.environ.get(
            "TRAIN_ORDER_REMAINING",
            "0",
        ).lower() not in {"0", "false", "no"},
    },
    "validation": {
        "include": optional_string_list("VALIDATION_INCLUDE_SOURCES"),
        "order": optional_string_list("VALIDATION_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("VALIDATION_EXCLUDE_SOURCES"),
        "order_remaining": os.environ.get(
            "VALIDATION_ORDER_REMAINING",
            "0",
        ).lower() not in {"0", "false", "no"},
    },
    "test": {
        "include": optional_string_list("TEST_INCLUDE_SOURCES"),
        "order": optional_string_list("TEST_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("TEST_EXCLUDE_SOURCES"),
        "order_remaining": os.environ.get(
            "TEST_ORDER_REMAINING",
            "0",
        ).lower() not in {"0", "false", "no"},
    },
}
for split_name in SPLIT_NAMES:
    for candidate_kind in ("missing", "existing"):
        options = CANDIDATE_SELECTION_OPTIONS[split_name][candidate_kind]
        range_configured = (
            options["min_idx"] is not None
            or options["max_idx"] is not None
        )
        take_name = f"{split_name.upper()}_{candidate_kind.upper()}_TAKE"
        if range_configured and take_name not in os.environ:
            options["take"] = None
        elif range_configured and options["take"] is not None:
            raise ValueError(
                f"{split_name} {candidate_kind}: configure either {take_name} "
                f"or {split_name.upper()}_{candidate_kind.upper()}_MIN_IDX/"
                f"{split_name.upper()}_{candidate_kind.upper()}_MAX_IDX, not both"
            )
DATASET_WORKERS = max(1, int(os.environ.get("DATASET_NUM_PROC", "8")))
DATASET_NUM_PROC = DATASET_WORKERS if DATASET_WORKERS > 1 else None
KEEP_IN_MEMORY = os.environ.get("KEEP_IN_MEMORY", "1").lower() not in {
    "0",
    "false",
    "no",
}

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "7680"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "64"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_LORA_RANK = int(os.environ.get("MAX_LORA_RANK", "32"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
ENABLE_PREFIX_CACHING = os.environ.get("ENABLE_PREFIX_CACHING", "1").lower() not in {
    "0",
    "false",
    "no",
}
ENABLE_CHUNKED_PREFILL = os.environ.get("ENABLE_CHUNKED_PREFILL", "1").lower() not in {
    "0",
    "false",
    "no",
}
CACHE_MODEL_WEIGHTS = os.environ.get("CACHE_MODEL_WEIGHTS", "0").lower() not in {
    "0",
    "false",
    "no",
}
CACHE_MODEL_WORKERS = int(os.environ.get("CACHE_MODEL_WORKERS", "16"))
CACHE_MODEL_CHUNK_MB = int(os.environ.get("CACHE_MODEL_CHUNK_MB", "1024"))
MATH_VERIFY_TIMEOUT_SECONDS = int(
    os.environ.get("MATH_VERIFY_TIMEOUT_SECONDS", "5")
)
if MATH_VERIFY_TIMEOUT_SECONDS <= 0:
    raise ValueError("MATH_VERIFY_TIMEOUT_SECONDS must be positive")

UPLOAD_TO_HF = os.environ.get("UPLOAD_TO_HF", "0").lower() not in {
    "0",
    "false",
    "no",
}
UPLOAD_TO_KAGGLE = os.environ.get("UPLOAD_TO_KAGGLE", "0").lower() not in {
    "0",
    "false",
    "no",
}
DEBUG_CSV_BACKUP = os.environ.get("DEBUG_CSV_BACKUP", "1").lower() in {
    "1",
    "true",
    "yes",
}
BACKUP_TRAJECTORIES = os.environ.get("BACKUP_TRAJECTORIES", "1").lower() not in {
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

BACKUP_FIELDS = (
    "split",
    "row_index",
    "id",
    "prompt",
    "stored_answer",
    "had_response",
    "chosen",
    "rejected",
    "trajectory_outputs",
    "trajectory_answers",
    "trajectory_correct",
)


# %%
import math_verify

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
    if example.get("source") in HQ_SOURCES:
        return True
    answer_type = clean_text(example.get("answer_type"))
    if answer_type is not None and answer_type.lower() in HQ_ANSWER_TYPES:
        return True
    return False
    # final_answer = clean_text(example.get("final_answer"))
    # return final_answer is not None and final_answer.isalnum()


def extract_competition_boxed_answer(text: Any) -> Optional[str]:
    if not text:
        return None
    text = str(text)
    boxed_starts = list(BOXED_START_RE.finditer(text))
    matches = []
    for index, match in enumerate(boxed_starts):
        start = match.end()
        end = (
            boxed_starts[index + 1].start()
            if index + 1 < len(boxed_starts)
            else len(text)
        )
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [match.strip() for match in matches if match.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return None


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
    if non_empty:
        return non_empty[-1]
    return spans[-1][2].strip()


def extract_fallback_answer(text: Optional[str]) -> str:
    if text is None:
        return "NOT_FOUND"

    for pattern in FALLBACK_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()

    matches = NUMBER_RE.findall(text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def extract_final_answers(text: Optional[str]) -> list[str]:
    answers = [
        extract_competition_boxed_answer(text),
        extract_balanced_boxed_answer(text),
    ]
    unique_answers = list(
        dict.fromkeys(answer for answer in answers if clean_text(answer) is not None)
    )
    if unique_answers:
        return unique_answers
    return [extract_fallback_answer(text)]


def extract_final_answer(text: Optional[str]) -> str:
    return extract_final_answers(text)[0]


def verify(stored_answer: Any, predicted: Any) -> bool:
    stored = clean_text(stored_answer)
    prediction = clean_text(predicted)
    if not stored:
        return not prediction or prediction == "NOT_FOUND"

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
    return f"<think>\n{normalized_reasoning}\n</think>\n{normalized_response}"


def split_think_content(value: Any) -> tuple[Optional[str], Optional[str]]:
    text = clean_text(value)
    if text is None:
        return None, None

    matches = list(THINK_RE.finditer(text))
    if not matches:
        return None, text

    reasoning = "\n\n".join(
        match.group(1).strip()
        for match in matches
        if match.group(1).strip()
    )
    response = THINK_RE.sub("", text).strip()
    return clean_text(reasoning), clean_text(response)


def select_preference(
    example: dict[str, Any],
    trajectory_outputs: list[str],
) -> dict[str, Any]:
    stored_answer = clean_text(example.get("final_answer"))
    extracted_answers = [
        extract_final_answers(output)
        for output in trajectory_outputs
    ]
    correctness = [
        any(verify(stored_answer, answer) for answer in answers)
        for answers in extracted_answers
    ]

    indexed_outputs = list(enumerate(trajectory_outputs))
    correct_outputs = [
        (index, output)
        for index, output in indexed_outputs
        if correctness[index]
    ]
    wrong_outputs = [
        (index, output)
        for index, output in indexed_outputs
        if not correctness[index]
    ]
    correct_outputs.sort(key=lambda item: (len(item[1]), item[0]))
    wrong_outputs.sort(key=lambda item: (len(item[1]), item[0]))

    existing_chosen = combine_reasoning_response(
        example.get("reasoning"),
        example.get("response"),
    )
    # chosen = (
    #     existing_chosen
    #     if existing_chosen is not None
    #     else correct_outputs[0][1] if correct_outputs else None
    # )
    chosen = (
        correct_outputs[0][1]
        if correct_outputs
        else existing_chosen if existing_chosen is not None else None
    )
    rejected = wrong_outputs[0][1] if wrong_outputs else None
    return {
        "chosen": chosen,
        "rejected": rejected,
        "trajectory_answers": extracted_answers,
        "trajectory_correct": correctness,
    }


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
        loaded = DatasetDict({"train": loaded})
    resumed_splits = {}
    for split_name, split_dataset in loaded.items():
        snapshot = incremental_split_path(split_name)
        if snapshot.exists():
            split_dataset = load_dataset(
                "parquet",
                data_files=str(snapshot),
                split="train",
                cache_dir=str(HF_CACHE_DIR),
                keep_in_memory=KEEP_IN_MEMORY,
            )
            print(f"{split_name}: resumed incremental snapshot {snapshot}")
        resumed_splits[split_name] = ensure_preference_columns(
            split_dataset,
            split_name,
        )
    return DatasetDict(resumed_splits)


def ensure_preference_columns(dataset, split_name: str):
    from datasets import Features, Value

    features = dict(dataset.features)
    features["chosen"] = Value("string")
    features["rejected"] = Value("string")
    features["dpo_row_index"] = Value("int64")
    features["dpo_selected"] = Value("bool")
    features["dpo_processed"] = Value("bool")

    def normalize_state(example: dict[str, Any], index: int) -> dict[str, Any]:
        row_index = example.get("dpo_row_index")
        return {
            "chosen": clean_text(example.get("chosen")),
            "rejected": clean_text(example.get("rejected")),
            "dpo_row_index": index if row_index is None else int(row_index),
            "dpo_selected": bool(example.get("dpo_selected", False)),
            "dpo_processed": bool(example.get("dpo_processed", False)),
        }

    return dataset.map(
        normalize_state,
        with_indices=True,
        features=Features(features),
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: initialize DPO columns",
        keep_in_memory=KEEP_IN_MEMORY,
    )


def incremental_split_path(split_name: str) -> Path:
    return INCREMENTAL_DIR / f"{split_name}.parquet"


def persist_incremental_split(dataset, split_name: str) -> None:
    INCREMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    target = incremental_split_path(split_name)
    temporary = target.with_suffix(".tmp.parquet")
    dataset.to_parquet(temporary)
    os.replace(temporary, target)
    print(f"{split_name}: persisted incremental snapshot to {target}")


def randomly_take_indices(
    indices: list[int],
    split_name: str,
    candidate_kind: str,
    purpose: str,
) -> list[int]:
    take_n = CANDIDATE_SELECTION_OPTIONS[split_name][candidate_kind]["take"]
    if take_n is None or take_n >= len(indices):
        return indices
    if take_n == 0:
        print(
            f"{split_name}: {split_name.upper()}_{candidate_kind.upper()}_"
            f"TAKE=0; skipping {candidate_kind} candidates"
        )
        return []

    import random

    random_generator = random.Random(
        f"{SEED}:{split_name}:{candidate_kind}:{purpose}"
    )
    return random_generator.sample(indices, take_n)


def order_indices_by_source(
    dataset,
    indices: list[int],
    split_name: str,
    order_remaining: Optional[bool] = None,
) -> list[int]:
    options = SOURCE_OPTIONS[split_name]
    include_sources = options["include"]
    order_sources = options["order"]
    exclude_sources = set(options["exclude"])
    if order_remaining is None:
        order_remaining = options.get("order_remaining", False)
    if not include_sources and not order_sources and not exclude_sources:
        return sorted(indices)
    if "source" not in dataset.column_names:
        raise KeyError(
            f"{split_name}: source controls require a 'source' column"
        )

    indices = [
        index
        for index in indices
        if dataset[index].get("source") not in exclude_sources
    ]
    dataset_sources = {
        source
        for source in dataset["source"]
        if source not in exclude_sources
    }
    missing_includes = [
        source for source in include_sources if source not in dataset_sources
    ]
    if missing_includes:
        raise ValueError(
            f"{split_name}: required INCLUDE_SOURCES are unavailable in the "
            f"split: {missing_includes}"
        )
    available_sources = list(
        dict.fromkeys(dataset[index].get("source") for index in indices)
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
    for index in indices:
        indices_by_source.setdefault(
            dataset[index].get("source"),
            [],
        ).append(index)
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
        remaining_indices = sorted(
            index
            for index in indices
            if dataset[index].get("source") in remaining_source_set
        )
    after_remaining_indices = [
        index
        for source in after_remaining_sources
        for index in indices_by_source.get(source, [])
    ]
    ordered_indices = (
        before_remaining_indices
        + remaining_indices
        + after_remaining_indices
    )
    print(
        f"{split_name}: candidate source order={ordered_sources}; "
        f"order_remaining={order_remaining}"
    )
    return ordered_indices


def finalize_candidate_indices(
    indices: list[int],
    split_name: str,
    candidate_kind: str,
    purpose: str,
) -> list[int]:
    options = CANDIDATE_SELECTION_OPTIONS[split_name][candidate_kind]
    min_idx = options["min_idx"]
    max_idx = options["max_idx"]
    if min_idx is not None or max_idx is not None:
        start = 0 if min_idx is None else min_idx
        stop = len(indices) if max_idx is None else min(max_idx, len(indices))
        if start > stop:
            raise ValueError(
                f"{split_name}: min index {start} exceeds max index {stop}"
            )
        selected = indices[start:stop]
        print(
            f"{split_name}: selected {candidate_kind} range "
            f"[{start:,}, {stop:,}) ({len(selected):,} rows)"
        )
        return selected
    return randomly_take_indices(
        indices,
        split_name,
        candidate_kind,
        purpose,
    )


def filter_high_quality_indices(
    dataset,
    indices: list[int],
    split_name: str,
    candidate_kind: str,
) -> list[int]:
    if not CANDIDATE_SELECTION_OPTIONS[split_name][candidate_kind]["filter_hq"]:
        return indices
    selected = [
        index
        for index in indices
        if is_high_quality_example(dataset[index])
    ]
    print(
        f"{split_name}: {candidate_kind} HQ filter retained "
        f"{len(selected):,}/{len(indices):,} candidates"
    )
    return selected


def select_generation_candidates(dataset, split_name: str):
    pending_selected_indices = [
        index
        for index, example in enumerate(dataset)
        if bool(example.get("dpo_selected"))
        and not bool(example.get("dpo_processed"))
    ]
    if pending_selected_indices:
        ordered_pending_indices = order_indices_by_source(
            dataset,
            pending_selected_indices,
            split_name,
        )
        selected_pending = dataset.select(
            ordered_pending_indices,
            keep_in_memory=KEEP_IN_MEMORY,
        )
        print(
            f"{split_name}: recovered {len(selected_pending):,}/"
            f"{len(pending_selected_indices):,} selected unprocessed candidates"
        )
        return dataset, selected_pending

    unprocessed_indices = [
        index
        for index, example in enumerate(dataset)
        if not example.get("dpo_processed")
    ]
    ordered_unprocessed_indices = order_indices_by_source(
        dataset,
        unprocessed_indices,
        split_name,
    )
    all_missing_response_indices = [
        index
        for index in ordered_unprocessed_indices
        if clean_text(dataset[index].get("response")) is None
    ]
    all_existing_response_indices = [
        index
        for index in ordered_unprocessed_indices
        if clean_text(dataset[index].get("response")) is not None
    ]
    all_missing_response_indices = filter_high_quality_indices(
        dataset,
        all_missing_response_indices,
        split_name,
        "missing",
    )
    all_existing_response_indices = filter_high_quality_indices(
        dataset,
        all_existing_response_indices,
        split_name,
        "existing",
    )
    finalized_missing_indices = finalize_candidate_indices(
        all_missing_response_indices,
        split_name,
        "missing",
        "new",
    )
    finalized_existing_indices = finalize_candidate_indices(
        all_existing_response_indices,
        split_name,
        "existing",
        "new",
    )
    merged_finalized_indices = order_indices_by_source(
        dataset,
        finalized_missing_indices + finalized_existing_indices,
        split_name,
    )
    finalized_candidate_index_set = set(merged_finalized_indices)
    if merged_finalized_indices:
        dataset = dataset.map(
            lambda example, index: {
                "dpo_selected": (
                    example.get("dpo_selected")
                    or index in finalized_candidate_index_set
                )
            },
            with_indices=True,
            num_proc=DATASET_NUM_PROC,
            desc=f"{split_name}: mark finalized candidates",
            keep_in_memory=KEEP_IN_MEMORY,
        )
        persist_incremental_split(dataset, split_name)

    candidates = (
        dataset.select(
            merged_finalized_indices,
            keep_in_memory=KEEP_IN_MEMORY,
        )
        if merged_finalized_indices
        else dataset.select([], keep_in_memory=KEEP_IN_MEMORY)
    )
    print(
        f"{split_name}: candidates={len(candidates):,}/"
        f"{len(ordered_unprocessed_indices):,} "
        f"(missing={len(finalized_missing_indices):,}/"
        f"{len(all_missing_response_indices):,}, "
        f"existing={len(finalized_existing_indices):,}/"
        f"{len(all_existing_response_indices):,})"
    )
    return dataset, candidates


# %%
def cache_model(
    path: str | Path,
    num_workers: Optional[int] = None,
    chunk_mb: int = 256,
) -> int:
    model_path = Path(path)
    files = sorted(
        file
        for file in model_path.rglob("*")
        if file.is_file()
        and file.suffix in {".bin", ".pt", ".safetensors"}
    )
    if not files:
        print(f"No model weight files found to cache at {model_path}")
        return 0

    workers = num_workers or min(multiprocessing.cpu_count(), 8)
    chunk_size = chunk_mb * 1024 * 1024

    def warm_file(file: Path) -> int:
        total = 0
        with file.open("rb") as handle:
            while data := handle.read(chunk_size):
                total += len(data)
        return total

    print(f"Caching {len(files)} model files with {workers} workers")
    started = time.time()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(warm_file, file) for file in files]
        for future in as_completed(futures):
            total_bytes += future.result()
    elapsed = time.time() - started
    print(
        f"Cached {total_bytes / 1024**3:.2f} GiB in {elapsed:.2f}s"
    )
    return total_bytes


def build_vllm_engine(
        model_path: Path | str = MODEL_PATH,
        tensor_parallel_size: int = TENSOR_PARALLEL_SIZE,
        max_num_seqs: int = MAX_NUM_SEQS,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
        max_model_len: int = MAX_MODEL_LEN,
        max_lora_rank: int = MAX_LORA_RANK,
        enable_prefix_caching: bool = ENABLE_PREFIX_CACHING,
        enable_chunked_prefill: bool = ENABLE_CHUNKED_PREFILL,
        seed: int = SEED,
        n: int = TRAJECTORIES,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        max_tokens: int = MAX_TOKENS,
        lora_path: Optional[Path | str] = LORA_PATH,
        cache_model_weights: bool = CACHE_MODEL_WEIGHTS,
        cache_model_workers: int = CACHE_MODEL_WORKERS,
        cache_model_chunk_mb: int = CACHE_MODEL_CHUNK_MB,
):
    if cache_model_weights:
        cache_model(
            model_path,
            num_workers=cache_model_workers,
            chunk_mb=cache_model_chunk_mb,
        )

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        seed=seed,
    )
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    lora_request = LoRARequest("adapter", 1, str(lora_path)) if lora_path else None
    return llm, sampling_params, lora_request


def format_generation_prompts(tokenizer: Any, prompts: list[str]) -> list[str]:
    formatted = []
    for prompt in prompts:
        user_content = f"{prompt}{BOXED_ANSWER_INSTRUCTION}"
        messages = [{"role": "user", "content": user_content}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered = user_content
        formatted.append(rendered)
    return formatted


# %%
def backup_path(split_name: str) -> Path:
    return BACKUP_DIR / f"{split_name}.csv"


def append_backup_records(split_name: str, records: list[dict[str, Any]]) -> None:
    if not DEBUG_CSV_BACKUP or not records:
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    path = backup_path(split_name)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BACKUP_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())


def apply_batch_updates(
    dataset,
    updates: dict[int, dict[str, Any]],
    split_name: str,
):
    def apply_update(example: dict[str, Any]) -> dict[str, Any]:
        row_index = int(example["dpo_row_index"])
        update = updates.get(row_index)
        if update is None:
            return {}
        return update

    return dataset.map(
        apply_update,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: apply generated preferences",
        keep_in_memory=KEEP_IN_MEMORY,
    )


def process_split(
    split_name: str,
    split_dataset,
    candidates,
    llm: Any,
    sampling_params: Any,
    lora_request: Any,
):
    from tqdm.auto import tqdm

    if not len(candidates):
        print(f"{split_name}: all candidates already processed")
        return split_dataset

    tokenizer = llm.get_tokenizer()
    progress = tqdm(
        range(0, len(candidates), PROMPT_BATCH_SIZE),
        desc=f"{split_name}: vLLM batches",
    )
    for offset in progress:
        batch = candidates.select(
            range(offset, min(offset + PROMPT_BATCH_SIZE, len(candidates))),
            keep_in_memory=KEEP_IN_MEMORY,
        )
        prompts = [str(prompt) for prompt in batch["prompt"]]
        formatted_prompts = format_generation_prompts(tokenizer, prompts)
        generated = llm.generate(
            formatted_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        if len(generated) != len(batch):
            raise RuntimeError(
                f"{split_name}: vLLM returned {len(generated)} outputs "
                f"for {len(batch)} prompts"
            )

        backup_rows = []
        updates = {}
        for example, request_output in zip(batch, generated):
            trajectory_outputs = [
                candidate.text
                for candidate in request_output.outputs
            ]
            if len(trajectory_outputs) != TRAJECTORIES:
                raise RuntimeError(
                    f"{split_name} row {example['dpo_row_index']}: expected "
                    f"{TRAJECTORIES} trajectories, received "
                    f"{len(trajectory_outputs)}"
                )
            preference = select_preference(example, trajectory_outputs)
            row_index = int(example["dpo_row_index"])
            update = {
                "chosen": preference["chosen"],
                "rejected": preference["rejected"],
                "dpo_selected": True,
                "dpo_processed": True,
            }
            if clean_text(example.get("response")) is None:
                generated_reasoning, generated_response = split_think_content(
                    preference["chosen"]
                )
                update["reasoning"] = generated_reasoning
                update["response"] = generated_response
            updates[row_index] = update
            backup_rows.append(
                {
                    "split": split_name,
                    "row_index": row_index,
                    "id": clean_text(example.get("id")) or "",
                    "prompt": clean_text(example.get("prompt")) or "",
                    "stored_answer": clean_text(example.get("final_answer")) or "",
                    "had_response": clean_text(example.get("response")) is not None,
                    "chosen": preference["chosen"] or "",
                    "rejected": preference["rejected"] or "",
                    "trajectory_outputs": json.dumps(
                        trajectory_outputs if BACKUP_TRAJECTORIES else [],
                        ensure_ascii=False,
                    ),
                    "trajectory_answers": json.dumps(
                        preference["trajectory_answers"],
                        ensure_ascii=False,
                    ),
                    "trajectory_correct": json.dumps(
                        preference["trajectory_correct"],
                    ),
                }
            )
        split_dataset = apply_batch_updates(
            split_dataset,
            updates,
            split_name,
        )
        persist_incremental_split(split_dataset, split_name)
        append_backup_records(split_name, backup_rows)
        processed_count = sum(split_dataset["dpo_processed"])
        progress.set_postfix(processed=processed_count)
    return split_dataset


def save_and_upload(
        dataset_dict, 
        upload_to_hf: bool = UPLOAD_TO_HF,
        upload_to_kaggle: bool = UPLOAD_TO_KAGGLE, 
        save_to_disk: bool = True,
) -> Any:
    if isinstance(dataset_dict, (str, Path)):
        from datasets import load_from_disk
        dataset_path = dataset_dict
        dataset_dict = load_from_disk(dataset_path, keep_in_memory=KEEP_IN_MEMORY)
        print(f"Dataset loaded from {dataset_path}")

    if save_to_disk:
        LOCAL_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(
            str(LOCAL_OUTPUT_DIR),
            num_proc=DATASET_NUM_PROC,
        )
        print(f"Saved updated dataset to {LOCAL_OUTPUT_DIR}")

    if upload_to_hf:
        try:
            if not HF_KEY:
                raise RuntimeError(
                    "UPLOAD_TO_HF=1 but HF_KEY/HF_TOKEN is not configured"
                )
            dataset_dict.push_to_hub(
                f"{HF_UPLOAD_USERNAME}/{DATASET_TAG}",
                private=True,
                token=HF_KEY,
            )
        except Exception as error:
            print(f"Upload to Hugging Face failed: {error}")
        else:
            print(f"Upload to Hugging Face succeeded")

    if upload_to_kaggle:
        try:
            if not KAGGLE_USERNAME or not KAGGLE_KEY:
                raise RuntimeError(
                    "UPLOAD_TO_KAGGLE=1 but KAGGLE_USERNAME/KAGGLE_KEY is not configured"
                )
            import kagglehub

            KAGGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for split_name, split_dataset in dataset_dict.items():
                split_dataset.to_parquet(
                    KAGGLE_OUTPUT_DIR / f"{split_name}.parquet"
                )
            kagglehub.dataset_upload(
                handle=KAGGLE_DATASET_REPO,
                local_dataset_dir=str(KAGGLE_OUTPUT_DIR),
            )

            (WORKING_DIR / "state.db").unlink(missing_ok=True)
            kagglehub.dataset_upload(
                handle=f"{KAGGLE_USERNAME}/{DATASET_TAG}-kaggle",
                local_dataset_dir=WORKING_DIR,
                version_notes=f"nemotron update dataset",
            )
        except Exception as error:
            print(f"Upload to Kaggle failed: {error}")
        else:
            print(f"Upload to Kaggle succeeded")

    return dataset_dict


# %%
model_path = Path(MODEL_PATH)
lora_path = Path(LORA_PATH) if LORA_PATH else LORA_PATH
if not model_path.exists():
    raise FileNotFoundError(f"Base model path does not exist: {model_path}")
if lora_path and not (lora_path / "adapter_config.json").exists():
    raise FileNotFoundError(
        f"LoRA adapter_config.json does not exist under {lora_path}"
    )

# %%
dataset_dict = load_reasoning_dataset()
available_splits = [
    split_name
    for split_name in SPLIT_NAMES
    if split_name in dataset_dict
]
if not available_splits:
    raise ValueError(
        f"None of {SPLIT_NAMES} exist in dataset: {list(dataset_dict)}"
    )

# %%
candidates_by_split = {}
for split_name in available_splits:
    updated_split, candidates = select_generation_candidates(
        dataset_dict[split_name],
        split_name,
    )
    dataset_dict[split_name] = updated_split
    candidates_by_split[split_name] = candidates

# %%
has_candidates = any(
    len(candidates)
    for candidates in candidates_by_split.values()
)
if has_candidates:
    llm, sampling_params, lora_request = build_vllm_engine()
else:
    llm = sampling_params = lora_request = None
    print("No prompts require preference generation")

# %%
if has_candidates:
    for split_name in available_splits:
        candidates = candidates_by_split[split_name]
        if not len(candidates):
            continue
        dataset_dict[split_name] = process_split(
            split_name,
            dataset_dict[split_name],
            candidates,
            llm,
            sampling_params,
            lora_request,
        )

# %%
save_and_upload(dataset_dict)
# dataset_dict = save_and_upload(LOCAL_OUTPUT_DIR, save_to_disk=False)

# %%
# from datasets import load_from_disk
# dataset_dict = load_from_disk(LOCAL_OUTPUT_DIR, keep_in_memory=KEEP_IN_MEMORY)

for split in SPLIT_NAMES:
    if ("chosen" not in dataset_dict[split].column_names 
        or "rejected" not in dataset_dict[split].column_names):
        continue

    valid_samples_1 = dataset_dict[split].filter(
        lambda x: x["chosen"] is not None 
        and x["rejected"] is not None 
        and len(x["chosen"]) > 0
        and len(x["rejected"]) > 0,
        keep_in_memory=KEEP_IN_MEMORY,
        desc=f"{split}: find non-empty `chosen` and `rejected`",
    )

    valid_samples_2 = dataset_dict[split].filter(
        lambda x: x["chosen"] is not None 
        and (x["rejected"] is None or len(x["rejected"]) == 0)
        and len(x["chosen"]) > 0,
        keep_in_memory=KEEP_IN_MEMORY,
        desc=f"{split}: find non-empty `chosen` only`",
    )
    
    valid_samples_3 = dataset_dict[split].filter(
        lambda x: x["rejected"] is not None 
        and (x["chosen"] is None or len(x["chosen"]) == 0)
        and len(x["rejected"]) > 0,
        keep_in_memory=KEEP_IN_MEMORY,
        desc=f"{split}: find non-empty `rejected` only",
    )
    
    print(
        f"{split}: Chosen and Rejected = {len(valid_samples_1)}, "
        f"Chosen Only = {len(valid_samples_2)}, "
        f"Rejected Only = {len(valid_samples_3)}"
    )
