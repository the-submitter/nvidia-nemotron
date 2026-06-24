from __future__ import annotations

import math
import os
import re
from typing import Any, Optional

import ray
import torch
from nemo_rl.data.processors import register_processor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import register_env

BOXED_START_RE = re.compile(r"\\boxed\{")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
FALLBACK_ANSWER_PATTERNS = [
    re.compile(r"The final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
]
EXACT_MATCH_WEIGHT = float(os.environ.get("EXACT_MATCH_WEIGHT", "5.0"))
ANSWER_FUZZY_WEIGHT = float(os.environ.get("ANSWER_FUZZY_WEIGHT", "3.0"))
COMPLETION_FUZZY_WEIGHT = float(os.environ.get("COMPLETION_FUZZY_WEIGHT", "0.15"))
BOXED_WEIGHT = float(os.environ.get("BOXED_WEIGHT", "1.0"))
THINK_WEIGHT = float(os.environ.get("THINK_WEIGHT", "0.25"))
NEMO_PROCESSOR_NAME = os.environ.get("NEMO_PROCESSOR_NAME", "nemotron_grpo_data_processor")
NEMO_ENV_NAME = os.environ.get("NEMO_ENV_NAME", "nemotron_unified_reward")


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def extract_boxed_answers_metric(text: Any) -> list[str]:
    if not text:
        return []
    value = str(text)
    boxed_starts = list(BOXED_START_RE.finditer(value))
    matches = []
    for index, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[index + 1].start() if index + 1 < len(boxed_starts) else len(value)
        segment = value[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    return [match.strip() for match in matches if match.strip()]


def extract_boxed_answers_balanced(text: Any) -> list[str]:
    if not text:
        return []
    value = str(text)
    spans = []
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
            spans.append(value[content_start : index - 1].strip())
            cursor = index
        else:
            cursor = content_start
    return [span for span in spans if span]


def extract_fallback_answer(text: Any) -> Optional[str]:
    if not text:
        return None
    value = str(text).strip()
    for pattern in FALLBACK_ANSWER_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1).strip().rstrip(".")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1].rstrip(".") if lines else None


def extract_final_answers(text: Any) -> list[str]:
    answers = []
    for answer in extract_boxed_answers_metric(text) + extract_boxed_answers_balanced(text):
        if answer not in answers:
            answers.append(answer)
    if not answers:
        fallback = extract_fallback_answer(text)
        if fallback:
            answers.append(fallback)
    return answers


def extract_final_answer(text: Any) -> Optional[str]:
    answers = extract_final_answers(text)
    return answers[-1] if answers else None


def verify(stored_answer: Any, predicted: Any) -> bool:
    expected = clean_text(stored_answer)
    actual = clean_text(predicted)
    if expected is None or actual is None:
        return False
    try:
        if math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-9):
            return True
    except Exception:
        pass
    try:
        from math_verify import parse, verify as math_verify_verify

        if math_verify_verify(parse(expected), parse(actual)):
            return True
    except Exception:
        pass
    return expected.strip().lower() == actual.strip().lower()


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        text = completion
    elif isinstance(completion, dict):
        text = str(completion.get("content") or "")
    elif isinstance(completion, list):
        text = "".join(completion_text(item) for item in completion)
    else:
        text = str(completion or "")
    if "</think>" in text.lower() and "<think" not in text.lower():
        text = "<think>\n" + text
    return text


def normalized_ratio(reference: Any, candidate: Any) -> float:
    ref = clean_text(reference)
    cand = clean_text(candidate)
    if ref is None or cand is None:
        return 0.0
    try:
        from rapidfuzz import fuzz, utils

        ref_processed = utils.default_process(ref) or ref
        cand_processed = utils.default_process(cand) or cand
        return max(
            fuzz.ratio(ref_processed, cand_processed),
            fuzz.token_set_ratio(ref_processed, cand_processed),
        ) / 100.0
    except Exception:
        return 1.0 if ref.strip().lower() == cand.strip().lower() else 0.0


def normalized_token_set_score(reference: Any, candidate: Any) -> float:
    ref = clean_text(reference)
    cand = clean_text(candidate)
    if ref is None or cand is None:
        return 0.0
    try:
        from rapidfuzz import fuzz, utils

        ref_processed = utils.default_process(ref) or ref
        cand_processed = utils.default_process(cand) or cand
        return fuzz.token_set_ratio(ref_processed, cand_processed) / 100.0
    except Exception:
        return 1.0 if ref.strip().lower() == cand.strip().lower() else 0.0


def combine_reasoning_response(reasoning: Any, response: Any) -> str:
    reasoning_text = clean_text(reasoning)
    response_text = clean_text(response)
    if reasoning_text and response_text:
        return f"<think>\n{reasoning_text}\n</think>\n{response_text}"
    return response_text or reasoning_text or ""


def unified_reward(prompts, completions, response, reasoning, final_answer, **kwargs):
    scores = []
    for completion, reference_response, reference_reasoning, answer in zip(
        completions, response, reasoning, final_answer
    ):
        text = completion_text(completion)
        boxed_answers = extract_boxed_answers_metric(text) + extract_boxed_answers_balanced(text)
        extracted_answers = boxed_answers or extract_final_answers(text)
        exact_score = max(
            (1.0 if verify(answer, candidate) else 0.0 for candidate in extracted_answers),
            default=0.0,
        )
        answer_fuzzy_score = max(
            (normalized_ratio(answer, candidate) for candidate in extracted_answers),
            default=0.0,
        )
        boxed_score = 1.0 if boxed_answers else 0.0
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


def nemotron_grpo_data_processor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx):
    prompt = str(datum_dict.get("input") or "")
    messages = [{"role": "user", "content": prompt}]
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
        "message_log": [{"role": "user", "content": rendered_prompt, "token_ids": token_ids}],
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
        "task_name": datum_dict.get("task_name"),
    }


def _assistant_completion_from_log(message_log):
    assistant_chunks = [
        str(message.get("content") or "")
        for message in message_log
        if message.get("role") == "assistant"
    ]
    if assistant_chunks:
        return "".join(assistant_chunks)
    return str(message_log[-1].get("content") or "") if message_log else ""


def _metadata_list(metadata, batch_size):
    if metadata is None:
        return [{} for _ in range(batch_size)]
    if isinstance(metadata, list):
        return [item if isinstance(item, dict) else {} for item in metadata]
    if isinstance(metadata, dict):
        return [metadata for _ in range(batch_size)]
    return [{} for _ in range(batch_size)]


@ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=-1)
class NemotronUnifiedRewardEnvironment(EnvironmentInterface[dict[str, Any]]):
    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def step(self, message_log_batch, metadata=None):
        completions = [
            _assistant_completion_from_log(message_log)
            for message_log in message_log_batch
        ]
        metadata_batch = _metadata_list(metadata, len(completions))
        scores = unified_reward(
            prompts=[str(item.get("prompt") or "") for item in metadata_batch],
            completions=completions,
            response=[item.get("response") for item in metadata_batch],
            reasoning=[item.get("reasoning") for item in metadata_batch],
            final_answer=[item.get("final_answer") for item in metadata_batch],
        )
        rewards = torch.tensor(scores, dtype=torch.float32)
        return EnvironmentReturn(
            observations=[
                {"role": "environment", "content": f"Reward: {float(score):.4f}"}
                for score in scores
            ],
            metadata=metadata_batch,
            next_stop_strings=[None for _ in scores],
            rewards=rewards,
            terminateds=torch.ones(len(scores), dtype=torch.bool),
            answers=[extract_final_answer(completion) for completion in completions],
        )

    def global_post_process_and_metrics(self, batch: BatchedDataDict):
        rewards = batch.get("rewards") if hasattr(batch, "get") else None
        if rewards is None:
            return batch, {}
        reward_tensor = rewards.detach().float().cpu()
        return batch, {
            "reward/mean": float(reward_tensor.mean().item()),
            "reward/max": float(reward_tensor.max().item()),
            "reward/min": float(reward_tensor.min().item()),
            "reward/exactish_rate": float(
                (reward_tensor >= EXACT_MATCH_WEIGHT).float().mean().item()
            ),
        }


def register_nemotron_components():
    try:
        register_processor(NEMO_PROCESSOR_NAME, nemotron_grpo_data_processor)
    except Exception as exc:
        if "already" not in str(exc).lower():
            raise
    try:
        register_env(
            NEMO_ENV_NAME,
            "nemotron_nemo_bridge.NemotronUnifiedRewardEnvironment",
        )
    except Exception as exc:
        if "already" not in str(exc).lower():
            raise


register_nemotron_components()
