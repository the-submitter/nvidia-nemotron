from __future__ import annotations

import math
import os
import re
from typing import Any, Optional

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

MATH_VERIFY_TIMEOUT_SECONDS = int(os.environ.get("MATH_VERIFY_TIMEOUT_SECONDS", "5"))
EXACT_MATCH_WEIGHT = float(os.environ.get("EXACT_MATCH_WEIGHT", "5.0"))
ANSWER_FUZZY_WEIGHT = float(os.environ.get("ANSWER_FUZZY_WEIGHT", "3.0"))
COMPLETION_FUZZY_WEIGHT = float(os.environ.get("COMPLETION_FUZZY_WEIGHT", "0.15"))
BOXED_WEIGHT = float(os.environ.get("BOXED_WEIGHT", "1.0"))
THINK_WEIGHT = float(os.environ.get("THINK_WEIGHT", "0.25"))


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_competition_boxed_answer(text: Any) -> Optional[str]:
    if not text:
        return None
    value = str(text)
    boxed_starts = list(BOXED_START_RE.finditer(value))
    matches = []
    for index, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[index + 1].start() if index + 1 < len(boxed_starts) else len(value)
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
    non_empty = [answer.strip() for _start, _end, answer in spans if answer.strip()]
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
    return extract_fallback_answer(text)


def extract_final_answer(text: Any) -> Optional[str]:
    answers = extract_final_answers(text)
    if isinstance(answers, list):
        non_empty = [answer for answer in answers if clean_text(answer) is not None]
        return non_empty[-1] if non_empty else None
    return answers


def verify(stored_answer: Any, predicted: Any) -> bool:
    stored = clean_text(stored_answer)
    prediction = clean_text(predicted)
    if not stored:
        return not prediction
    if prediction is None:
        return False

    if BINARY_RE.fullmatch(stored):
        return prediction.casefold() == stored.casefold()

    try:
        if math.isclose(float(stored), float(prediction), rel_tol=1e-2, abs_tol=1e-5):
            return True
    except Exception:
        pass

    try:
        import math_verify

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


def combine_reasoning_response(reasoning: Any, response: Any) -> str:
    normalized_response = clean_text(response)
    normalized_reasoning = clean_text(reasoning)
    if normalized_reasoning is not None and normalized_response is not None:
        return f"<think>\n{normalized_reasoning}\n</think>\n{normalized_response}"
    return normalized_response or normalized_reasoning or ""


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        text = completion
    elif isinstance(completion, dict):
        text = str(completion.get("content") or "")
    elif isinstance(completion, list):
        text = "".join(completion_text(item) for item in completion)
    else:
        try:
            text = str(completion[0].get("content", ""))
        except Exception:
            text = str(completion or "")
    if "</think>" in text.casefold() and "<think" not in text.casefold():
        text = f"<think>\n{text}"
    return text


def normalized_fuzzy_score(gold: Any, target: Any) -> float:
    if gold is None:
        return 0.0
    try:
        from rapidfuzz import fuzz, utils

        gold_text = utils.default_process(clean_text(gold) or "")
        target_text = utils.default_process(clean_text(target) or "")
        return fuzz.ratio(gold_text, target_text) / 100.0
    except Exception:
        return 1.0 if clean_text(gold) == clean_text(target) else 0.0


def normalized_token_set_score(gold: Any, target: Any) -> float:
    if gold is None:
        return 0.0
    try:
        from rapidfuzz import fuzz, utils

        gold_text = utils.default_process(clean_text(gold) or "")
        target_text = utils.default_process(clean_text(target) or "")
        return fuzz.token_set_ratio(gold_text, target_text) / 100.0
    except Exception:
        return 1.0 if clean_text(gold) == clean_text(target) else 0.0


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
        extracted_answers = extract_final_answers(text)

        if isinstance(extracted_answers, list):
            boxed_answers = extracted_answers
        else:
            extracted_answers = [extracted_answers]
            boxed_answers = [None]

        exact_score = max(
            (1.0 if verify(target, extracted_answer) else 0.0 for extracted_answer in extracted_answers),
            default=0.0,
        )
        answer_fuzzy_score = max(
            (normalized_fuzzy_score(target, extracted_answer) for extracted_answer in extracted_answers),
            default=0.0,
        )
        boxed_score = max(
            (1.0 if clean_text(extracted_answer) is not None else 0.0 for extracted_answer in boxed_answers),
            default=0.0,
        )
        reference_completion = combine_reasoning_response(reference_reasoning, reference_response)
        completion_fuzzy_score = normalized_token_set_score(reference_completion, text)
        think_matches = [match.group(1).strip() for match in THINK_RE.finditer(text)]
        think_score = 1.0 if any(think_matches) else 0.0

        scores.append(
            EXACT_MATCH_WEIGHT * exact_score
            + ANSWER_FUZZY_WEIGHT * answer_fuzzy_score
            + COMPLETION_FUZZY_WEIGHT * completion_fuzzy_score
            + BOXED_WEIGHT * boxed_score
            + THINK_WEIGHT * think_score
        )
    return scores
