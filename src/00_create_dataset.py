# %%
from __future__ import annotations

import heapq
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional

# %%
# User secrets
try:
    from kaggle_secrets import UserSecretsClient  # type: ignore

    user_secrets = UserSecretsClient()
    KAGGLE_KEY = user_secrets.get_secret("KAGGLE_KEY")
    KAGGLE_USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
    HF_KEY = user_secrets.get_secret("HF_KEY")
except Exception:
    KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
    KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
    HF_KEY = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")

if KAGGLE_KEY:
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
if KAGGLE_USERNAME:
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
if HF_KEY:
    os.environ["HF_TOKEN"] = HF_KEY

# %%
# %%capture
# !pip install -qqq "datasets>=5.0.0" kagglehub tqdm --root-user-action ignore

# %%
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    IterableDataset,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    VerificationMode,
)
from tqdm import tqdm

# MAX_WORKERS = max(1, int(os.environ.get("MAX_WORKERS", os.cpu_count() or 1)))
MAX_WORKERS = 1
print(f"MAX_WORKERS: {MAX_WORKERS}")
NUM_PROC = MAX_WORKERS if MAX_WORKERS > 1 else None
STREAM_LOG_INTERVAL = int(os.environ.get("STREAM_LOG_INTERVAL", "1000"))
KEEP_IN_MEMORY = os.environ.get("KEEP_IN_MEMORY", "1").lower() not in {"0", "false", "no"}
HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", "/tmp/hf_cache"))
KAGGLE_CACHE_DIR = Path(os.environ.get("KAGGLE_CACHE_DIR", "/tmp/kagglehub"))
WORKING_DIR = Path(os.environ.get("WORKING_DIR", "/kaggle/working"))
LOCAL_COMPETITION_PATH = Path(
    os.environ.get(
        "LOCAL_COMPETITION_PATH",
        "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge",
    )
)
LOCAL_NEMOTRON_COT_TONG_PATH = Path(
    os.environ.get(
        "LOCAL_NEMOTRON_COT_TONG_PATH",
        "/kaggle/input/datasets/dgxchen/nemotron-cot-tong",
    )
)
HF_UPLOAD_USERNAME = os.environ.get("HF_UPLOAD_USERNAME", "the-submitter")
DATASET_TAG = os.environ.get("DATASET_TAG", "nemotron-reasoning")
LOCAL_OUTPUT_DIR = WORKING_DIR / DATASET_TAG
UPLOAD_TO_HF = os.environ.get("UPLOAD_TO_HF", "1").lower() not in {"0", "false", "no"}
UPLOAD_TO_KAGGLE = os.environ.get("UPLOAD_TO_KAGGLE", "1").lower() not in {
    "0",
    "false",
    "no",
}
FILTER_HQ_BY_SPLIT = {
    "train": os.environ.get("TRAIN_FILTER_HQ", "0").lower() not in {"0", "false", "no"},
    "validation": os.environ.get("VALIDATION_FILTER_HQ", "0").lower() not in {"0", "false", "no"},
    "test": os.environ.get("TEST_FILTER_HQ", "0").lower() not in {"0", "false", "no"},
}
GLOBAL_DEDUPE_BY_SPLIT = {
    "train": os.environ.get("TRAIN_GLOBAL_DEDUPE", "1").lower() not in {"0", "false", "no"},
    "validation": os.environ.get("VALIDATION_GLOBAL_DEDUPE", "1").lower() not in {"0", "false", "no"},
    "test": os.environ.get("TEST_GLOBAL_DEDUPE", "1").lower() not in {"0", "false", "no"},
}
SHUFFLE_BY_SPLIT = {
    "train": os.environ.get("TRAIN_SHUFFLE", "1").lower() not in {"0", "false", "no"},
    "validation": os.environ.get("VALIDATION_SHUFFLE", "1").lower() not in {"0", "false", "no"},
    "test": os.environ.get("TEST_SHUFFLE", "1").lower() not in {"0", "false", "no"},
}

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))

if KEEP_IN_MEMORY:
    import datasets

    datasets.config.IN_MEMORY_MAX_SIZE = int(
        os.environ.get("IN_MEMORY_MAX_SIZE", str(30 * 1024**3))
    )

# %%
SPLIT_NAMES = ("train", "validation", "test")
SCHEMA_COLUMNS = (
    "id",
    "source",
    "domain",
    "prompt",
    "reasoning",
    "response",
    "final_answer",
    "answer_type",
    "difficulty",
)
SCHEMA_FEATURES = Features({column: Value("string") for column in SCHEMA_COLUMNS})

Processor = Callable[[dict[str, Any]], Optional[dict[str, Any]]]
LengthGetter = Callable[[dict[str, Any]], Any]
ExamplePredicate = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    quotas: Mapping[str, int]
    processor: Processor
    subset: Optional[str] = None
    source: Literal["huggingface", "local"] = "huggingface"
    split: Optional[str] = "train"
    filter_key: Optional[str] = None
    length_getter: Optional[LengthGetter] = None
    filter_by_length: Optional[Literal["max", "min"]] = None
    filter_map: Mapping[str, set[Any]] = field(default_factory=dict)
    filter_map_include: bool = True
    stream_filter: Optional[ExamplePredicate] = None
    dedupe: bool = True
    shuffle_seed: int = 42
    local_files: tuple[str, ...] = ()
    load_kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_size(self) -> int:
        return sum(self.quotas.get(split, 0) for split in SPLIT_NAMES)


# %%
URL_RE = re.compile(r"https?://", re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
GSM8K_ANSWER_RE = re.compile(r"^####\s*(.+?)\s*$")
BOXED_START_RE = re.compile(r"\\boxed\{")
INTEGER_RE = re.compile(r"^[+-]?\d[\d,]*$")
FLOAT_RE = re.compile(r"^[+-]?(?:\d[\d,]*\.\d+|\d[\d,]*[eE][+-]?\d+)$")
FRACTION_RE = re.compile(r"^[+-]?(?:\d+\s*/\s*\d+|\\frac\s*\{.+?\}\s*\{.+?\})$")
MULTIPLE_CHOICE_RE = re.compile(r"^(?:[A-Ea-e])$")
LEVEL_RE = re.compile(r"\b([1-5])\b")

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


def contains_url(value: Any) -> bool:
    return bool(value and URL_RE.search(str(value)))


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


# def extract_boxed_spans(text: Any) -> list[tuple[int, int, str]]:
#     if not text:
#         return []

#     value = str(text)
#     boxed_starts = list(BOXED_START_RE.finditer(value))
#     spans: list[tuple[int, int, str]] = []
#     for index, match in enumerate(boxed_starts):
#         content_start = match.end()
#         segment_end = (
#             boxed_starts[index + 1].start()
#             if index + 1 < len(boxed_starts)
#             else len(value)
#         )
#         segment = value[content_start:segment_end]
#         last_brace = segment.rfind("}")
#         if last_brace == -1:
#             spans.append((match.start(), segment_end, segment))
#         else:
#             spans.append(
#                 (
#                     match.start(),
#                     content_start + last_brace + 1,
#                     segment[:last_brace],
#                 )
#             )
#     return spans


def strip_boxed(value: Any) -> Optional[str]:
    text = clean_text(value)
    if text is None:
        return None
    spans = extract_boxed_spans(text)
    if spans and spans[-1][0] == 0 and spans[-1][1] == len(text):
        return clean_text(spans[-1][2])
    return text


def extract_last_boxed_answer(value: Any) -> Optional[str]:
    spans = extract_boxed_spans(value)
    if not spans:
        return None
    non_empty = [
        answer.strip()
        for _, _, answer in spans
        if answer.strip()
    ]
    if non_empty:
        return non_empty[-1]
    return spans[-1][2].strip()


def reconcile_final_answer(response: Any, final_answer: Any) -> Optional[str]:
    normalized_answer = strip_boxed(final_answer)
    response_answer = extract_last_boxed_answer(response)
    if response_answer is not None and response_answer != normalized_answer:
        return response_answer
    return normalized_answer


def split_think_content(value: Any) -> tuple[Optional[str], Optional[str]]:
    text = clean_text(value)
    if text is None:
        return None, None

    matches = list(THINK_RE.finditer(text))
    if not matches:
        return None, text

    reasoning = "\n\n".join(
        match.group(1).strip() for match in matches if match.group(1).strip()
    )
    response = THINK_RE.sub("", text).strip()
    return clean_text(reasoning), clean_text(response)


def assistant_message_content(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    assistant_contents = [
        clean_text(message.get("content"))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    return next((content for content in reversed(assistant_contents) if content), None)


def normalize_domain(value: Any, default: str = "other") -> str:
    text = (clean_text(value) or default).lower()
    if "proof" in text:
        return "proof"
    if any(token in text for token in ("logic", "deduct", "folio", "boolean")):
        return "logic"
    if any(token in text for token in ("puzzle", "zebra", "enigmata", "game")):
        return "puzzle"
    if any(token in text for token in ("arith", "number", "counting")):
        return "arithmetic"
    if any(
        token in text
        for token in (
            "math",
            "algebra",
            "geometry",
            "calculus",
            "probability",
            "statistics",
            "gcd",
            "lcm",
            "equation",
        )
    ):
        return "math"
    return default


def infer_answer_type(value: Any, multiple_choice: bool = False) -> str:
    answer = strip_boxed(value)
    if answer is None:
        return "text"
    compact = answer.strip()
    if multiple_choice or MULTIPLE_CHOICE_RE.fullmatch(compact):
        return "multiple_choice"
    if INTEGER_RE.fullmatch(compact):
        return "integer"
    if FLOAT_RE.fullmatch(compact):
        return "float"
    if FRACTION_RE.fullmatch(compact):
        return "fraction"
    if any(token in compact for token in ("\\", "^", "_", "=", "+", "*", "(", ")")):
        return "expression"
    return "text"


def normalize_difficulty(value: Any) -> str:
    text = (clean_text(value) or "").lower()
    level_match = LEVEL_RE.search(text)
    if level_match:
        level = int(level_match.group(1))
        return "easy" if level <= 2 else "medium" if level == 3 else "hard"
    if any(token in text for token in ("easy", "simple", "basic")):
        return "easy"
    if any(token in text for token in ("medium", "intermediate")):
        return "medium"
    if any(token in text for token in ("hard", "advanced", "difficult")):
        return "hard"
    return "unknown"


def normalized_record(
    *,
    source: str,
    prompt: Any,
    final_answer: Any,
    record_id: Any = None,
    domain: Any = "other",
    reasoning: Any = None,
    response: Any = None,
    answer_type: Optional[str] = None,
    difficulty: Any = None,
    multiple_choice: bool = False,
    reconcile_answer: bool = True,
) -> Optional[dict[str, Any]]:
    normalized_prompt = clean_text(prompt)
    if normalized_prompt is None:
        return None

    normalized_response = clean_text(response)
    normalized_answer = (
        reconcile_final_answer(normalized_response, final_answer)
        if reconcile_answer
        else strip_boxed(final_answer)
    )
    return {
        "id": clean_text(record_id),
        "source": source,
        "domain": normalize_domain(domain),
        "prompt": normalized_prompt,
        "reasoning": clean_text(reasoning),
        "response": normalized_response,
        "final_answer": normalized_answer,
        "answer_type": answer_type
        or infer_answer_type(normalized_answer, multiple_choice=multiple_choice),
        "difficulty": normalize_difficulty(difficulty),
    }


def stable_id(source: str, split: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")
    return f"{slug}-{split}-{index:07d}"


def assistant_content_length(example: dict[str, Any]) -> int:
    return len(assistant_message_content(example.get("messages")) or "")


# %%
def process_numina(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    answer = clean_text(example.get("answer"))
    if (
        answer is None
        or "proof" in answer.lower()
        or "z" in answer.lower()
        or "=" in answer
        or example.get("question_type") == "proof"
        or example.get("problem_is_valid") != "Yes"
        or example.get("solution_is_valid") != "Yes"
        or contains_url(example.get("problem"))
        or contains_url(example.get("solution"))
    ):
        return None

    solution = clean_text(example.get("solution"))
    if solution is None:
        return None
    response = f"{solution}\n\nFinal answer: \\boxed{{{answer}}}"
    return normalized_record(
        source="AI-MO/NuminaMath-1.5",
        record_id=example.get("id"),
        domain=example.get("question_type"),
        prompt=example.get("problem"),
        response=response,
        final_answer=answer,
        multiple_choice=example.get("question_type") == "MCQ",
    )


def process_competition_math(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    solution = clean_text(example.get("solution"))
    if solution is None or example.get("level") == "Level 1":
        return None
    spans = extract_boxed_spans(solution)
    if not spans:
        return None

    start, end, boxed_content = spans[-1]
    curated_answer = boxed_content.split("=")[-1].strip()
    if not curated_answer:
        return None
    curated_solution = f"{solution[:start]}\\boxed{{{curated_answer}}}{solution[end:]}"
    return normalized_record(
        source="qwedsacf/competition_math",
        domain=example.get("type"),
        prompt=example.get("problem"),
        response=curated_solution,
        final_answer=curated_answer,
        difficulty=example.get("level"),
    )


def process_openr1_math(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    if contains_url(example.get("problem")) or contains_url(example.get("solution")):
        return None
    content = assistant_message_content(example.get("messages"))
    if content is None:
        return None
    reasoning, response = split_think_content(content)
    return normalized_record(
        source="open-r1/OpenR1-Math-220k",
        record_id=example.get("uuid"),
        domain=example.get("question_type"),
        prompt=example.get("problem"),
        reasoning=reasoning,
        response=response,
        final_answer=example.get("answer"),
        multiple_choice=example.get("question_type") == "MCQ",
    )


def process_gsm8k(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    answer_text = clean_text(example.get("answer"))
    if answer_text is None:
        return None
    lines = answer_text.splitlines()
    match = GSM8K_ANSWER_RE.fullmatch(lines[-1].strip()) if lines else None
    if match is None:
        return None
    final_answer = match.group(1).strip()
    response = "\n".join(lines[:-1]).rstrip()
    response = f"{response}\nFinal answer: \\boxed{{{final_answer}}}".strip()
    return normalized_record(
        source="openai/gsm8k",
        domain="math",
        prompt=example.get("question"),
        response=response,
        final_answer=final_answer,
    )


def process_svamp(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    equation = clean_text(example.get("Equation"))
    answer = clean_text(example.get("Answer"))
    if equation is None or answer is None:
        return None
    prompt = example.get("question_concat") or " ".join(
        part
        for part in (clean_text(example.get("Body")), clean_text(example.get("Question")))
        if part
    )
    return normalized_record(
        source="ChilleD/SVAMP",
        record_id=example.get("ID"),
        domain=example.get("Type"),
        prompt=prompt,
        response=f"Equation: {equation}\nFinal answer: \\boxed{{{answer}}}",
        final_answer=answer,
    )


def process_asdiv(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    if example.get("solution_type") not in {"Geometry", "Algebra-2", "LCM", "GCD"}:
        return None
    prompt = " ".join(
        part
        for part in (clean_text(example.get("body")), clean_text(example.get("question")))
        if part
    )
    return normalized_record(
        source="EleutherAI/asdiv",
        domain=example.get("solution_type"),
        prompt=prompt,
        final_answer=example.get("answer"),
    )


def process_drop(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    answer = example.get("answer")
    if not isinstance(answer, dict):
        return None
    number = clean_text(answer.get("number"))
    spans = answer.get("spans") or []
    final_answer = number or clean_text(",".join(map(str, spans)))
    prompt = "\n\n".join(
        part
        for part in (clean_text(example.get("passage")), clean_text(example.get("question")))
        if part
    )
    return normalized_record(
        source="EleutherAI/drop",
        domain="math",
        prompt=prompt,
        final_answer=final_answer,
    )


def process_proofwriter(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    theory = clean_text(example.get("theory"))
    question = clean_text(example.get("question"))
    if theory is None or question is None:
        return None
    prompt = (
        f"{theory}\n\nStatement: {question}\n\n"
        "Is the statement right given the context? Answer as True, False or Unknown."
    )
    return normalized_record(
        source="tasksource/proofwriter",
        domain="math",
        prompt=prompt,
        final_answer=example.get("answer"),
    )


def process_folio(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    premises = clean_text(example.get("premises"))
    conclusion = clean_text(example.get("conclusion"))
    if premises is None or conclusion is None:
        return None
    prompt = (
        f"### Premises:\n{premises}\n\n### Conclusion:\n{conclusion}\n\n"
        "Is the conclusion right given the premises? Answer as True, False or Uncertain."
    )
    return normalized_record(
        source="yale-nlp/FOLIO",
        record_id=example.get("example_id") or example.get("story_id"),
        domain="logic",
        prompt=prompt,
        final_answer=example.get("label"),
    )


def process_prontoqa(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    context = clean_text(example.get("context"))
    question = clean_text(example.get("question"))
    answer = clean_text(example.get("answer"))
    options = example.get("options")
    if context is None or question is None or answer is None or not isinstance(options, list):
        return None

    resolved_answer = "uncertain"
    for option in options:
        option_text = clean_text(option)
        if option_text is None:
            continue
        option_parts = option_text.split()
        if option_parts and answer[0].lower() == option_parts[0][0].lower():
            resolved_answer = option_parts[-1].strip().lower()
            break

    return normalized_record(
        source="renma/ProntoQA",
        record_id=example.get("id"),
        domain="logic",
        prompt=f"### Context:\n{context}\n\n### Question: {question}",
        final_answer=resolved_answer,
        multiple_choice=True,
    )


def process_zebra_logic(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    puzzle = clean_text(example.get("puzzle"))
    question = clean_text(example.get("question"))
    if puzzle is None or question is None:
        return None
    return normalized_record(
        source="WildEval/ZebraLogic",
        record_id=example.get("id"),
        domain="puzzle",
        prompt=f"### Puzzle:\n{puzzle}\n\n### Question: {question}",
        final_answer=example.get("answer"),
        multiple_choice=True,
    )


def replace_enigmata_code_blocks(prompt: Any) -> Optional[str]:
    text = clean_text(prompt)
    if text is None:
        return None
    text = text.replace(
        "a code block (```)",
        r"boxed format (\boxed{...})",
    )
    return re.sub(
        r"```(?:[A-Za-z0-9_+-]+\s*\n)?(.*?)```",
        lambda match: rf"\boxed{{{match.group(1).strip()}}}",
        text,
        flags=re.DOTALL,
    )


def process_enigmata(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    return normalized_record(
        source="BytedTsinghua-SIA/Enigmata-Eval",
        record_id=example.get("id"),
        domain=example.get("task_type"),
        prompt=replace_enigmata_code_blocks(example.get("prompt")),
        final_answer=example.get("answer"),
        difficulty=example.get("ability"),
    )


def process_open_math_reasoning(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    reasoning, response = split_think_content(example.get("generated_solution"))
    return normalized_record(
        source="nvidia/OpenMathReasoning",
        domain="math",
        prompt=example.get("problem"),
        reasoning=reasoning,
        response=response,
        final_answer=example.get("expected_answer"),
        difficulty=example.get("pass_rate_72b_tir"),
    )


def process_nemotron_cot_tong(
    example: dict[str, Any],
) -> Optional[dict[str, Any]]:
    generated_cot = clean_text(example.get("generated_cot"))
    if generated_cot is not None and "</think>" in generated_cot:
        if not generated_cot.lstrip().startswith("<think>"):
            generated_cot = f"<think>\n{generated_cot}"
    reasoning, response = split_think_content(generated_cot)
    return normalized_record(
        source="dgxchen/nemotron-cot-tong",
        record_id=example.get("id"),
        domain=example.get("type"),
        prompt=example.get("prompt"),
        reasoning=reasoning,
        response=response,
        final_answer=example.get("answer"),
        reconcile_answer=False,
    )


def valid_open_math_reasoning(example: dict[str, Any]) -> bool:
    return (
        clean_text(example.get("problem")) is not None
        and clean_text(example.get("generated_solution")) is not None
    )


def process_competition_data(example: dict[str, Any]) -> Optional[dict[str, Any]]:
    return normalized_record(
        source="nvidia-nemotron-model-reasoning-challenge",
        record_id=example.get("id"),
        domain="puzzle",
        prompt=example.get("prompt"),
        final_answer=example.get("answer"),
    )


# %%
DATA_CONFIGS = [
    DatasetConfig(
        name="AI-MO/NuminaMath-1.5",
        quotas={"train": 15_400, "validation": 100, "test": 100},
        processor=process_numina,
        filter_key="solution",
        filter_by_length="min",
    ),
    DatasetConfig(
        name="qwedsacf/competition_math",
        quotas={"train": 7_700, "validation": 50, "test": 50},
        processor=process_competition_math,
        filter_key="solution",
        filter_by_length="min",
    ),
    DatasetConfig(
        name="open-r1/OpenR1-Math-220k",
        subset="default",
        quotas={"train": 15_400, "validation": 100, "test": 100},
        processor=process_openr1_math,
        length_getter=assistant_content_length,
        filter_by_length="min",
    ),
    DatasetConfig(
        name="openai/gsm8k",
        subset="main",
        quotas={"train": 3_000, "validation": 30, "test": 30},
        processor=process_gsm8k,
        filter_key="answer",
        filter_by_length="max",
    ),
    DatasetConfig(
        name="ChilleD/SVAMP",
        quotas={"train": 10, "validation": 2, "test": 2},
        processor=process_svamp,
        filter_key="Equation",
        filter_by_length="max",
    ),
    DatasetConfig(
        name="EleutherAI/asdiv",
        split="validation",
        quotas={"train": 30, "validation": 3, "test": 3},
        processor=process_asdiv,
        filter_key="formula",
        filter_by_length="max",
    ),
    DatasetConfig(
        name="EleutherAI/drop",
        quotas={"train": 15_400, "validation": 100, "test": 100},
        processor=process_drop,
        filter_key="passage",
        filter_by_length="min",
    ),
    DatasetConfig(
        name="tasksource/proofwriter",
        quotas={"train": 7_700, "validation": 50, "test": 50},
        processor=process_proofwriter,
    ),
    DatasetConfig(
        name="yale-nlp/FOLIO",
        quotas={"train": 980, "validation": 10, "test": 10},
        processor=process_folio,
    ),
    DatasetConfig(
        name="renma/ProntoQA",
        split="validation",
        quotas={"train": 100, "validation": 4, "test": 4},
        processor=process_prontoqa,
    ),
    DatasetConfig(
        name="WildEval/ZebraLogic",
        subset="mc_mode",
        split="test",
        quotas={"train": 1_890, "validation": 19, "test": 19},
        processor=process_zebra_logic,
        filter_key="puzzle",
        filter_by_length="min",
    ),
    DatasetConfig(
        name="BytedTsinghua-SIA/Enigmata-Eval",
        quotas={"train": 4_666, "validation": 46, "test": 46},
        processor=process_enigmata,
    ),
    DatasetConfig(
        name="nvidia/OpenMathReasoning",
        split="cot",
        quotas={"train": 18_110, "validation": 181, "test": 181},
        processor=process_open_math_reasoning,
        filter_key="generated_solution",
        filter_by_length="min",
        stream_filter=valid_open_math_reasoning,
        load_kwargs=dict(
            data_files={"cot": "data/cot-*.parquet"},
            verification_mode=VerificationMode.NO_CHECKS,
            streaming=True,
        ),
    ),
    DatasetConfig(
        name=str(LOCAL_NEMOTRON_COT_TONG_PATH),
        source="local",
        quotas={"train": 7_830, "validation": 0, "test": 0},
        processor=process_nemotron_cot_tong,
        local_files=("problem_ids_matched.csv",),
        dedupe=False,
    ),
    DatasetConfig(
        name=str(LOCAL_COMPETITION_PATH),
        source="local",
        quotas={"train": 9_500, "validation": 0, "test": 0},
        processor=process_competition_data,
    ),
]


# %%
def resolve_split(
    dataset: Dataset | DatasetDict | IterableDataset,
    requested_split: Optional[str],
) -> Dataset | IterableDataset:
    if isinstance(dataset, (Dataset, IterableDataset)):
        return dataset
    if requested_split:
        if requested_split not in dataset:
            raise KeyError(
                f"Requested split {requested_split!r}; available splits: {list(dataset)}"
            )
        return dataset[requested_split]
    if "train" in dataset:
        return dataset["train"]
    if len(dataset) == 1:
        return next(iter(dataset.values()))
    return concatenate_datasets([dataset[name] for name in dataset])


def load_local_dataset(
    path: Path,
    split: Optional[str],
    local_files: tuple[str, ...] = (),
) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {path}")

    if (path / "dataset_dict.json").exists() or (path / "dataset_info.json").exists():
        return resolve_split(
            load_from_disk(str(path), keep_in_memory=KEEP_IN_MEMORY),
            split,
        )

    extensions = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "json",
        ".parquet": "parquet",
    }
    if local_files:
        files = [path / file_name for file_name in local_files]
        missing_files = [file for file in files if not file.is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing configured local dataset files: {missing_files}"
            )
    else:
        files = [
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in extensions
        ]
    if not files:
        raise FileNotFoundError(f"No supported dataset files found under {path}")

    preferred_names = [split, "train"] if split else ["train"]
    selected_files = files
    for preferred_name in preferred_names:
        matches = [
            file
            for file in files
            if preferred_name and preferred_name.lower() in file.stem.lower()
        ]
        if matches:
            selected_files = matches
            break

    formats = {extensions[file.suffix.lower()] for file in selected_files}
    if len(formats) != 1:
        raise ValueError(f"Local dataset mixes file formats: {selected_files}")
    data_format = formats.pop()
    loaded = load_dataset(
        data_format,
        data_files=[str(file) for file in selected_files],
        split=split,
        cache_dir=str(HF_CACHE_DIR),
        keep_in_memory=KEEP_IN_MEMORY,
    )
    return loaded


def get_length_value(example: dict[str, Any], config: DatasetConfig) -> Any:
    if config.length_getter is not None:
        return config.length_getter(example)
    if config.filter_key is not None:
        return example.get(config.filter_key)
    return None


def matches_filter_map(example: dict[str, Any], config: DatasetConfig) -> bool:
    if not config.filter_map:
        return True
    matches = all(
        example.get(key) in accepted_values
        for key, accepted_values in config.filter_map.items()
    )
    return matches if config.filter_map_include else not matches


def materialize_streaming_dataset(
    dataset: IterableDataset,
    config: DatasetConfig,
) -> Dataset:
    if config.filter_by_length is None:
        raise ValueError(
            f"{config.name}: streaming materialization requires filter_by_length "
            "so the bounded in-memory selector knows which rows to retain"
        )
    if config.filter_key is None and config.length_getter is None:
        raise ValueError(
            f"{config.name}: streaming materialization requires filter_key "
            "or length_getter"
        )

    target_size = config.total_size
    retained: list[tuple[int, int, int, dict[str, Any]]] = []
    seen = 0
    eligible = 0
    stream_ds = tqdm(enumerate(dataset), desc=f"{config.name}: materialize dataset stream")

    for index, example in stream_ds:
        seen += 1
        if STREAM_LOG_INTERVAL and seen % STREAM_LOG_INTERVAL == 0:
            stream_ds.set_postfix(
                eligible=f"{eligible:,}", 
                retained=f"{len(retained):,}",
            )

        if not matches_filter_map(example, config):
            continue
        value = get_length_value(example, config)
        if value is None:
            continue
        if config.stream_filter is not None:
            if not config.stream_filter(example):
                continue
        elif config.processor(example) is None:
            continue

        eligible += 1
        length = value if isinstance(value, int) else len(value)
        score = -length if config.filter_by_length == "min" else length
        entry = (score, -index, index, example)

        if len(retained) < target_size:
            heapq.heappush(retained, entry)
        elif entry[:2] > retained[0][:2]:
            heapq.heapreplace(retained, entry)

    if len(retained) < target_size:
        raise ValueError(
            f"{config.name}: requested {target_size:,} valid streamed records, "
            f"but only {len(retained):,} were available from {seen:,} rows"
        )

    retained.sort(key=lambda item: (-item[0], item[2]))
    rows = [item[3] for item in retained]
    print(
        f"{config.name}: materialized {len(rows):,} ranked rows in RAM "
        f"from {seen:,} streamed rows"
    )
    return Dataset.from_list(rows, features=dataset.features)


def load_source_dataset(config: DatasetConfig) -> Dataset:
    if config.source == "local":
        return load_local_dataset(
            Path(config.name),
            config.split,
            config.local_files,
        )

    load_kwargs = dict(config.load_kwargs)
    streaming = bool(load_kwargs.get("streaming"))
    load_kwargs.update(
        {
            "split": config.split,
            "token": HF_KEY,
            "cache_dir": str(HF_CACHE_DIR),
        }
    )
    if not streaming:
        load_kwargs["keep_in_memory"] = KEEP_IN_MEMORY
    if config.split is None:
        load_kwargs.pop("split")
    dataset = load_dataset(config.name, config.subset, **load_kwargs)
    dataset = resolve_split(dataset, config.split)
    if isinstance(dataset, IterableDataset):
        return materialize_streaming_dataset(dataset, config)
    return dataset


def filter_dataset_by_map(dataset: Dataset, config: DatasetConfig) -> Dataset:
    if not config.filter_map:
        return dataset

    return dataset.filter(
        lambda example: matches_filter_map(example, config),
        num_proc=NUM_PROC,
        desc=f"{config.name}: filter map",
        keep_in_memory=KEEP_IN_MEMORY,
    )


def process_dataset(dataset: Dataset, config: DatasetConfig) -> Dataset:
    original_columns = dataset.column_names
    processed = dataset.map(
        lambda example: config.processor(example) or {column: None for column in SCHEMA_COLUMNS},
        remove_columns=original_columns,
        features=SCHEMA_FEATURES,
        num_proc=NUM_PROC,
        desc=f"{config.name}: normalize",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    processed = processed.filter(
        lambda example: example["prompt"] is not None,
        num_proc=NUM_PROC,
        desc=f"{config.name}: remove invalid",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    if config.dedupe:
        processed = dedupe_dataset_by_prompt(processed, config.name)
    return processed


def dedupe_score(example: dict[str, Any]) -> int:
    response = clean_text(example.get("response"))
    if response is None:
        return -1
    return len(response) + len(clean_text(example.get("reasoning")) or "")


def select_deduped_prompt_indices(dataset: Dataset, desc: str) -> list[int]:
    selected_by_prompt: dict[str, tuple[int, int]] = {}
    dedupe_examples = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc=desc,
    )
    for index, example in dedupe_examples:
        prompt = example["prompt"]
        score = dedupe_score(example)
        if prompt in selected_by_prompt:
            _current_index, current_score = selected_by_prompt[prompt]
            if score < 0:
                continue
            if current_score >= 0 and current_score <= score:
                continue
        selected_by_prompt[prompt] = (index, score)

    return sorted(
        index for index, _score in selected_by_prompt.values()
    )


def dedupe_dataset_by_prompt(dataset: Dataset, name: str) -> Dataset:
    selected_indices = select_deduped_prompt_indices(
        dataset,
        desc=f"{name}: dedupe by `prompt`",
    )
    if len(selected_indices) == len(dataset):
        print(f"{name}: removed 0 duplicate prompts")
        return dataset

    deduped = dataset.select(
        selected_indices,
        keep_in_memory=KEEP_IN_MEMORY,
    )
    print(
        f"{name}: removed {len(dataset) - len(deduped):,} "
        "duplicate prompts"
    )
    return deduped


def filter_dataset_by_length(dataset: Dataset, config: DatasetConfig) -> Dataset:
    if (
        (config.filter_key is None and config.length_getter is None)
        or config.filter_by_length is None
    ):
        return dataset
    if config.filter_key is not None and config.filter_key not in dataset.column_names:
        raise KeyError(
            f"{config.name}: length key {config.filter_key!r} is unavailable"
        )

    dataset = dataset.filter(
        lambda example: get_length_value(example, config) is not None,
        num_proc=NUM_PROC,
        desc=f"{config.name}: non-empty length value",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    dataset = dataset.map(
        lambda example: {
            "__filter_length": (
                get_length_value(example, config)
                if isinstance(get_length_value(example, config), int)
                else len(get_length_value(example, config))
            )
        },
        num_proc=NUM_PROC,
        desc=f"{config.name}: measure length",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    dataset = dataset.sort(
        "__filter_length",
        reverse=config.filter_by_length == "max",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    return dataset.remove_columns("__filter_length")


def allocate_fixed_splits(dataset: Dataset, config: DatasetConfig) -> DatasetDict:
    if len(dataset) < config.total_size:
        raise ValueError(
            f"{config.name}: requested {config.total_size:,} valid records, "
            f"but only {len(dataset):,} are available"
        )

    selected = dataset.select(range(config.total_size), keep_in_memory=KEEP_IN_MEMORY)
    selected = selected.shuffle(seed=config.shuffle_seed, keep_in_memory=KEEP_IN_MEMORY)

    output = {}
    offset = 0
    for split in SPLIT_NAMES:
        size = config.quotas.get(split, 0)
        split_dataset = (
            selected.select(range(offset, offset + size), keep_in_memory=KEEP_IN_MEMORY)
            if size
            else selected.select([], keep_in_memory=KEEP_IN_MEMORY)
        )
        if size:
            split_dataset = split_dataset.map(
                lambda _example, index: {
                    "id": (
                        f"{stable_id(config.name, split, index)}-{_example['id']}"
                        if _example["id"]
                        else stable_id(config.name, split, index)
                    )
                },
                with_indices=True,
                num_proc=NUM_PROC,
                desc=f"{config.name}: assign {split} ids",
                keep_in_memory=KEEP_IN_MEMORY,
            )
        output[split] = split_dataset
        offset += size
    return DatasetDict(output)


def prepare_dataset(config: DatasetConfig) -> DatasetDict:
    print(f"\nPreparing {config.name} ({config.total_size:,} records)")
    dataset = load_source_dataset(config)
    dataset = filter_dataset_by_map(dataset, config)
    dataset = filter_dataset_by_length(dataset, config)
    dataset = process_dataset(dataset, config)
    result = allocate_fixed_splits(dataset, config)
    print({split: len(result[split]) for split in SPLIT_NAMES})
    return result


def build_final_dataset(configs: list[DatasetConfig]) -> DatasetDict:
    prepared = [prepare_dataset(config) for config in configs]
    final_splits = {}
    for split in SPLIT_NAMES:
        parts = [dataset[split] for dataset in prepared if len(dataset[split])]
        if not parts:
            final_splits[split] = Dataset.from_dict(
                {column: [] for column in SCHEMA_COLUMNS},
                features=SCHEMA_FEATURES,
            )
            continue
        combined = concatenate_datasets(parts)

        if GLOBAL_DEDUPE_BY_SPLIT.get(split):
            combined = dedupe_dataset_by_prompt(combined, f"Final `{split}`")

        if FILTER_HQ_BY_SPLIT.get(split):
            before_hq = len(combined)
            combined = combined.filter(
                is_high_quality_example,
                num_proc=NUM_PROC,
                desc=f"{split}: keep high-quality examples",
                keep_in_memory=KEEP_IN_MEMORY,
            )
            print(
                f"{split}: HQ filter retained "
                f"{len(combined):,}/{before_hq:,} examples"
            )

        if SHUFFLE_BY_SPLIT.get(split):
            combined = combined.shuffle(
                seed=configs[0].shuffle_seed,
                keep_in_memory=KEEP_IN_MEMORY,
            )
            print(
                f"{split}: shuffled {len(combined):,} examples "
                f"with seed {configs[0].shuffle_seed}"
            )

        final_splits[split] = combined
    return DatasetDict(final_splits)


def validate_final_dataset(
    dataset: DatasetDict,
    configs: list[DatasetConfig],
) -> None:
    expected_sizes = {
        split: sum(config.quotas.get(split, 0) for config in configs)
        for split in SPLIT_NAMES
    }
    actual_sizes = {split: len(dataset[split]) for split in SPLIT_NAMES}
    if any(actual_sizes[split] > expected_sizes[split] for split in SPLIT_NAMES):
        raise AssertionError(
            f"Split sizes exceed configured quotas: "
            f"expected_max={expected_sizes}, actual={actual_sizes}"
        )

    for split in SPLIT_NAMES:
        if tuple(dataset[split].column_names) != SCHEMA_COLUMNS:
            raise AssertionError(
                f"{split} schema differs: {dataset[split].column_names}"
            )
        if len(set(dataset[split]["id"])) != len(dataset[split]):
            raise AssertionError(f"{split} contains duplicate ids")
        if (
            GLOBAL_DEDUPE_BY_SPLIT[split]
            and len(set(dataset[split]["prompt"])) != len(dataset[split])
        ):
            raise AssertionError(f"{split} contains duplicate prompts")
    print("Validated split sizes:", actual_sizes)


# %%
final_dataset = build_final_dataset(DATA_CONFIGS)


# %%
validate_final_dataset(final_dataset, DATA_CONFIGS)
print(final_dataset)


# %%
final_dataset.save_to_disk(
    str(LOCAL_OUTPUT_DIR),
    num_proc=NUM_PROC,
)
print(f"Saved dataset to {LOCAL_OUTPUT_DIR}")


# %%
if UPLOAD_TO_HF:
    try:
        if not HF_KEY:
            raise RuntimeError("UPLOAD_TO_HF=1 but HF_KEY/HF_TOKEN is not configured")
        final_dataset.push_to_hub(
            f"{HF_UPLOAD_USERNAME}/{DATASET_TAG}",
            private=True,
            token=HF_KEY,
        )
    except Exception as e:
        print(f"Upload to HF failed: {e}")
    else:
        print(f"Upload to HF succeeded")


# %%
if UPLOAD_TO_KAGGLE:
    try:
        if not KAGGLE_USERNAME or not KAGGLE_KEY:
            raise RuntimeError(
                "UPLOAD_TO_KAGGLE=1 but KAGGLE_USERNAME/KAGGLE_KEY is not configured"
            )

        import kagglehub

        kaggle_folder = KAGGLE_CACHE_DIR / DATASET_TAG
        kaggle_folder.mkdir(parents=True, exist_ok=True)
        for split_name, split_dataset in final_dataset.items():
            split_dataset.to_parquet(kaggle_folder / f"{split_name}.parquet")

        kagglehub.dataset_upload(
            handle=f"{KAGGLE_USERNAME}/{DATASET_TAG}",
            local_dataset_dir=str(kaggle_folder),
        )
    except Exception as e:
        print(f"Upload to Kaggle failed: {e}")
    else:
        print(f"Upload to Kaggle succeeded")
