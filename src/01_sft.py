# %%
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# %%
# # User secrets
# try:
#     from kaggle_secrets import UserSecretsClient  # type: ignore

#     user_secrets = UserSecretsClient()
#     HF_KEY = user_secrets.get_secret("HF_KEY")
#     WANDB_KEY = user_secrets.get_secret("WANDB_KEY")
# except Exception:
#     HF_KEY = os.environ.get("HF_KEY") or os.environ.get("HF_TOKEN")
#     WANDB_KEY = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY")

# if HF_KEY:
#     os.environ["HF_TOKEN"] = HF_KEY
# if WANDB_KEY:
#     os.environ["WANDB_API_KEY"] = WANDB_KEY
HF_KEY = WANDB_KEY = None

# %%
wheels_dir = "/kaggle/input/datasets/rohitraje0493/unsloth-vllm-wheels/packages"
# !pip install uv --no-index --find-links={wheels_dir}
# !uv pip install \
#     "triton>=3.3.0" \
#     "torchvision==0.25.0+cu128" \
#     bitsandbytes \
#     "transformers>=4.56.2" \
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
# # !uv pip install vllm --no-index --find-links={wheels_dir}

# %%
WORKING_DIR = Path(os.environ.get("WORKING_DIR", "/kaggle/working"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/default/1",
)
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/kaggle/input/datasets/rohitraje0493/nemotron-reasoning",
)
DATASET_REVISION = os.environ.get("DATASET_REVISION")
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

TRAIN_MIN_IDX = optional_nonnegative_int("TRAIN_MIN_IDX")
TRAIN_MAX_IDX = optional_nonnegative_int("TRAIN_MAX_IDX")
EVAL_MIN_IDX = optional_nonnegative_int("EVAL_MIN_IDX", 15)
EVAL_MAX_IDX = optional_nonnegative_int("EVAL_MAX_IDX", 35)

LORA_STAGE = os.environ.get("LORA_STAGE", "sft")
LORA_VERSION = os.environ.get("LORA_VERSION", "v1")
RUN_NAME = os.environ.get("RUN_NAME", f"nemotron-{LORA_STAGE}-{LORA_VERSION}")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKING_DIR / RUN_NAME)))
ADAPTER_OUTPUT_DIR = Path(
    os.environ.get(
        "ADAPTER_OUTPUT_DIR",
        str(WORKING_DIR / f"nemotron-lora-{LORA_STAGE}-{LORA_VERSION}"),
    )
)
ADAPTER_INPUT_PATH = os.environ.get("ADAPTER_INPUT_PATH")
HF_USERNAME = os.environ.get("HF_USERNAME", "the-submitter")
HF_ADAPTER_REPO = os.environ.get(
    "HF_ADAPTER_REPO",
    f"{HF_USERNAME}/nemotron-lora-{LORA_STAGE}-{LORA_VERSION}",
)

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "8192"))
DATASET_WORKERS = max(1, int(os.environ.get("DATASET_NUM_PROC", "8")))
DATASET_NUM_PROC = DATASET_WORKERS if DATASET_WORKERS > 1 else None
SEED = int(os.environ.get("SEED", "3407"))

PER_DEVICE_TRAIN_BATCH_SIZE = int(
    os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "4")
)
PER_DEVICE_EVAL_BATCH_SIZE = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("GRADIENT_ACCUMULATION_STEPS", "8")
)
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "10"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "20"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(SAVE_STEPS)))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))

LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "64"))
REPORT_TO = os.environ.get("REPORT_TO", "wandb")
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT")
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "0").lower() not in {
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
INSTRUCTION_PART = os.environ.get(
    "INSTRUCTION_PART",
    "<|im_start|>user\n",
)
RESPONSE_PART = os.environ.get(
    "RESPONSE_PART",
    "<|im_start|>assistant\n",
)

if REPORT_TO == "wandb":
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "/kaggle/working/wandb_logs"
    os.makedirs("/kaggle/working/wandb_logs", exist_ok=True)


# %%
def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def is_trainable_example(example: dict[str, Any]) -> bool:
    return (
        clean_text(example.get("prompt")) is not None
        and clean_text(example.get("response")) is not None
    )


def build_user_content(prompt: Any) -> str:
    normalized_prompt = clean_text(prompt)
    if normalized_prompt is None:
        raise ValueError("Cannot build a conversation without a prompt")
    if r"\boxed" in normalized_prompt:
        return normalized_prompt
    return f"{normalized_prompt}\n\n{BOXED_ANSWER_INSTRUCTION}"


def build_conversation(example: dict[str, Any]) -> dict[str, Any]:
    response = clean_text(example.get("response"))
    if response is None:
        raise ValueError("Dataset must be filtered before conversation formatting")

    assistant_message = {
        "role": "assistant",
        "content": response,
    }
    reasoning = clean_text(example.get("reasoning"))
    if reasoning is not None:
        assistant_message["reasoning_content"] = reasoning

    return {
        "conversations": [
            {
                "role": "user",
                "content": build_user_content(example.get("prompt")),
            },
            assistant_message,
        ]
    }


def render_conversations(
    examples: dict[str, list[Any]],
    tokenizer: Any,
) -> dict[str, list[str]]:
    texts = []
    for conversation in examples["conversations"]:
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        assistant_message = conversation[-1]
        reasoning = clean_text(assistant_message.get("reasoning_content"))
        if reasoning is not None and reasoning not in text:
            fallback_conversation = [
                dict(message)
                for message in conversation
            ]
            fallback_assistant = dict(fallback_conversation[-1])
            fallback_assistant.pop("reasoning_content", None)
            fallback_assistant["content"] = (
                f"<think>{reasoning}</think>\n"
                f"{fallback_assistant['content']}"
            )
            fallback_conversation[-1] = fallback_assistant
            text = tokenizer.apply_chat_template(
                fallback_conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
        texts.append(text)
    return {"text": texts}


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
                keep_in_memory=KEEP_IN_MEMORY,
            )
    else:
        loaded = load_dataset(
            DATASET_PATH,
            revision=DATASET_REVISION,
            token=HF_KEY,
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


def prepare_split(dataset, tokenizer, split_name: str, already_filtered: bool = False):
    original_size = len(dataset)
    if not already_filtered:
        dataset = dataset.filter(
            is_trainable_example,
            num_proc=DATASET_NUM_PROC,
            desc=f"{split_name}: keep non-empty responses",
            keep_in_memory=KEEP_IN_MEMORY,
        )
    if not len(dataset):
        raise ValueError(f"{split_name} has no examples with non-empty responses")

    dataset = dataset.map(
        build_conversation,
        remove_columns=dataset.column_names,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: build conversations",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    conversation_features = dataset.features
    dataset = dataset.map(
        render_conversations,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: apply chat template",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    if dataset.features == conversation_features or "text" not in dataset.column_names:
        raise RuntimeError(f"{split_name}: chat-template formatting did not create text")

    print(f"{split_name}: retained {len(dataset):,}/{original_size:,} examples")
    return dataset


def select_index_range(
    dataset,
    min_idx: Optional[int],
    max_idx: Optional[int],
    split_name: str,
):
    if min_idx is None and max_idx is None:
        return dataset

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

    selected = dataset.select(range(start, stop))
    if not len(selected):
        raise ValueError(
            f"{split_name}: index range [{start}, {stop}) selected no examples"
        )
    print(
        f"{split_name}: selected [{start:,}, {stop:,}) "
        f"({len(selected):,} examples)"
    )
    return selected


def prepare_datasets(tokenizer):
    dataset_dict = load_reasoning_dataset()
    train_dataset = prepare_split(
        dataset_dict[TRAIN_SPLIT],
        tokenizer,
        TRAIN_SPLIT,
    )
    train_dataset = select_index_range(
        train_dataset,
        TRAIN_MIN_IDX,
        TRAIN_MAX_IDX,
        TRAIN_SPLIT,
    )

    eval_dataset = None
    if EVAL_SPLIT in dataset_dict and len(dataset_dict[EVAL_SPLIT]):
        filtered_eval = dataset_dict[EVAL_SPLIT].filter(
            is_trainable_example,
            num_proc=DATASET_NUM_PROC,
            desc=f"{EVAL_SPLIT}: check non-empty responses",
            keep_in_memory=KEEP_IN_MEMORY,
        )
        if len(filtered_eval):
            eval_dataset = prepare_split(
                filtered_eval,
                tokenizer,
                EVAL_SPLIT,
                already_filtered=True,
            )
            eval_dataset = select_index_range(
                eval_dataset,
                EVAL_MIN_IDX,
                EVAL_MAX_IDX,
                EVAL_SPLIT,
            )
    return train_dataset, eval_dataset


# %%
def load_model_and_tokenizer():
    from unsloth import FastLanguageModel

    model_source = ADAPTER_INPUT_PATH or MODEL_PATH
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        token=HF_KEY,
        use_gradient_checkpointing="unsloth",
        # unsloth_tiled_mlp=True,
        gpu_memory_utilization=0.95,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if ADAPTER_INPUT_PATH is None:
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


# %%
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from unsloth import FastLanguageModel  # noqa: F401
import torch
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

if not torch.cuda.is_available():
    raise RuntimeError("SFT training requires a CUDA GPU")

# %%
model, tokenizer = load_model_and_tokenizer()

# %%
train_dataset, eval_dataset = prepare_datasets(tokenizer)

# %%
has_eval = eval_dataset is not None and len(eval_dataset) > 0
bf16 = torch.cuda.is_bf16_supported()

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir=str(OUTPUT_DIR),
        run_name=RUN_NAME,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=DATASET_WORKERS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
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
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=SEED,
        data_seed=SEED,
        report_to=REPORT_TO,
        bf16=bf16,
        fp16=not bf16,
        tf32=True,
        padding_free=False,
        packing=False,
        packing_strategy="bfd",     # "wrapped"
        # remove_unused_columns=True,
        # max_grad_norm=0.5,
        # assistant_only_loss=True,
        # completion_only_loss=True,
        # dataset_kwargs = {"skip_prepare_dataset": True},
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part=INSTRUCTION_PART,
    response_part=RESPONSE_PART,
)

# %%
sample = trainer.train_dataset[0]
if not any(label != -100 for label in sample["labels"]):
    raise RuntimeError(
        "Response-only masking removed every label. Check INSTRUCTION_PART and "
        "RESPONSE_PART against the tokenizer chat template."
    )

gpu = torch.cuda.get_device_properties(0)
start_reserved = torch.cuda.max_memory_reserved() / 1024**3
print(
    f"GPU: {gpu.name}; VRAM={gpu.total_memory / 1024**3:.2f} GiB; "
    f"initial_reserved={start_reserved:.2f} GiB"
)
print("Rendered training sample:")
print(train_dataset[0]["text"][:2000])
print(tokenizer.decode(trainer.train_dataset[0]["input_ids"])[:2000])
print(tokenizer.decode(
        [tokenizer.pad_token_id if x == -100 else x 
        for x in trainer.train_dataset[0]["labels"]]
    ).replace(tokenizer.pad_token, " ")[:2000]
)

# %%
trainer_stats = trainer.train(
    resume_from_checkpoint=resolve_resume_from_checkpoint()
)

# %%
trainer.save_state()
model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))

# %%
if PUSH_TO_HUB:
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

# %%
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
runtime = trainer_stats.metrics.get("train_runtime", 0.0)
print(f"Training runtime: {runtime:.2f} seconds ({runtime / 60:.2f} minutes)")
print(f"Peak reserved VRAM: {peak_reserved:.2f} GiB")
print(f"Saved LoRA adapter to {ADAPTER_OUTPUT_DIR}")

# %%
# !zip -r submission.zip {ADAPTER_OUTPUT_DIR}
