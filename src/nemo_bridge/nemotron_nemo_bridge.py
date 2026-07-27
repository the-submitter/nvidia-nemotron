from __future__ import annotations

import os
from typing import Any

import ray
import torch
from nemo_rl.data.processors import register_processor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import register_env

from nemotron_reward_utils import (
    EXACT_MATCH_WEIGHT,
    extract_final_answer,
    unified_reward,
)

NEMO_PROCESSOR_NAME = "nemotron_grpo_data_processor"
NEMO_ENV_NAME = "nemotron_unified_reward"
NEMO_ENV_ACTOR_FQN = (
    "nemotron_nemo_bridge.NemotronUnifiedRewardEnvironment"
)


def _configure_single_gpu_bf16_lora() -> None:
    """Keep the frozen 30B base in BF16 so it fits on one 96 GB GPU.

    NeMo-RL normally retains FP32 master parameters, which is appropriate when
    the 30B model is sharded across the 16 GPUs in NVIDIA's reference recipe.
    A single-GPU LoRA run only updates adapter parameters and cannot hold the
    unsharded 30B base in FP32. This opt-in Kaggle patch keeps the frozen base
    in BF16 while retaining FP32 LoRA master parameters; FSDP2 casts the latter
    to BF16 for forward/backward compute.
    """
    if os.environ.get("NEMO_KAGGLE_BF16_LORA", "0") != "1":
        return

    import nemo_rl.models.automodel.setup as automodel_setup

    auto_config = automodel_setup.AutoConfig
    original_config_loader = auto_config.from_pretrained.__func__
    if not getattr(original_config_loader, "_nemotron_bf16_patched", False):
        def bf16_config_loader(cls, *args, **kwargs):
            if kwargs.get("torch_dtype") is torch.float32:
                kwargs["torch_dtype"] = torch.bfloat16
            return original_config_loader(cls, *args, **kwargs)

        bf16_config_loader._nemotron_bf16_patched = True
        auto_config.from_pretrained = classmethod(bf16_config_loader)

    peft_config_class = automodel_setup.PeftConfig
    original_peft_loader = peft_config_class.from_dict.__func__
    if not getattr(original_peft_loader, "_nemotron_bf16_patched", False):
        def fp32_peft_loader(cls, values):
            values = {**values, "lora_dtype": "torch.float32"}
            return original_peft_loader(cls, values)

        fp32_peft_loader._nemotron_bf16_patched = True
        peft_config_class.from_dict = classmethod(fp32_peft_loader)

    automodel_setup._disable_automodel_checkpoint_dtype_restore = lambda: None


def nemotron_grpo_data_processor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx):
    prompt = str(datum_dict.get("input") or "")
    if task_data_spec.prompt:
        prompt = task_data_spec.prompt.format(prompt)
    messages = [{"role": "user", "content": prompt}]
    if task_data_spec.system_prompt:
        messages.insert(
            0,
            {"role": "system", "content": task_data_spec.system_prompt},
        )
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
    if max_seq_length is not None and token_ids.numel() >= max_seq_length:
        token_ids = token_ids[: min(4, max_seq_length)]
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
        "loss_multiplier": loss_multiplier,
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
        if len(metadata) != batch_size:
            raise ValueError(
                f"Metadata batch has {len(metadata)} items for {batch_size} completions"
            )
        return [item if isinstance(item, dict) else {} for item in metadata]
    if isinstance(metadata, dict):
        return [metadata for _ in range(batch_size)]
    return [{} for _ in range(batch_size)]


@ray.remote(num_cpus=1, max_restarts=-1, max_task_retries=-1)
class NemotronUnifiedRewardEnvironment(EnvironmentInterface[dict[str, Any]]):
    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def step(
        self,
        message_log_batch,
        metadata=None,
        return_extracted_answer: bool = False,
    ):
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
            answers=(
                [extract_final_answer(completion) for completion in completions]
                if return_extracted_answer
                else None
            ),
        )

    def global_post_process_and_metrics(self, batch: BatchedDataDict):
        rewards = batch.get("rewards") if hasattr(batch, "get") else None
        if rewards is None and hasattr(batch, "get"):
            rewards = batch.get("total_reward")
        if rewards is None:
            return batch, {}
        reward_tensor = rewards.detach().float().cpu()
        if hasattr(batch, "get") and batch.get("is_end") is not None:
            reward_tensor = reward_tensor * batch["is_end"].detach().float().cpu()
        return batch, {
            "reward/mean": float(reward_tensor.mean().item()),
            "reward/max": float(reward_tensor.max().item()),
            "reward/min": float(reward_tensor.min().item()),
            "reward/exactish_rate": float(
                (reward_tensor >= EXACT_MATCH_WEIGHT).float().mean().item()
            ),
        }


def register_nemotron_components():
    # Custom environments must be present in both NeMo registries. Kaggle uses
    # one pre-populated offline environment for the driver and every Ray actor.
    if os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM", "0") == "1":
        for actor_fqn in tuple(ACTOR_ENVIRONMENT_REGISTRY):
            ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] = PY_EXECUTABLES.SYSTEM
        _configure_single_gpu_bf16_lora()
    ACTOR_ENVIRONMENT_REGISTRY[NEMO_ENV_ACTOR_FQN] = PY_EXECUTABLES.SYSTEM

    try:
        register_processor(NEMO_PROCESSOR_NAME, nemotron_grpo_data_processor)
    except Exception as exc:
        if "already" not in str(exc).lower():
            raise
    try:
        register_env(
            NEMO_ENV_NAME,
            NEMO_ENV_ACTOR_FQN,
        )
    except Exception as exc:
        if "already" not in str(exc).lower():
            raise


register_nemotron_components()
