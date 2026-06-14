# %%
from __future__ import annotations

import json
import os
import shutil
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
# # !uv pip install vllm --no-index --find-links={wheels_dir}
# # !uv pip install "protobuf<6.0.0" --no-index --find-links={wheels_dir}

# %%
WORKING_DIR = Path(os.environ.get("WORKING_DIR", "/kaggle/working"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    # "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/default/1",
)
BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
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


TRAIN_MIN_IDX = optional_nonnegative_int("TRAIN_MIN_IDX", 0)
TRAIN_MAX_IDX = optional_nonnegative_int("TRAIN_MAX_IDX", 7830)
EVAL_MIN_IDX = optional_nonnegative_int("EVAL_MIN_IDX", 15)
EVAL_MAX_IDX = optional_nonnegative_int("EVAL_MAX_IDX", 35)
SOURCE_OPTIONS = {
    TRAIN_SPLIT: {
        "include": optional_string_list("TRAIN_INCLUDE_SOURCES"),
        "order": optional_string_list("TRAIN_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("TRAIN_EXCLUDE_SOURCES"),
    },
    EVAL_SPLIT: {
        "include": optional_string_list("EVAL_INCLUDE_SOURCES"),
        "order": optional_string_list("EVAL_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("EVAL_EXCLUDE_SOURCES"),
    },
}
TRAIN_SHUFFLE = os.environ.get("TRAIN_SHUFFLE", "1").lower() not in {
    "0",
    "false",
    "no",
}
EVAL_SHUFFLE = os.environ.get("EVAL_SHUFFLE", "1").lower() not in {
    "0",
    "false",
    "no",
}

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
KAGGLE_ADAPTER_REPO = os.environ.get(
    "KAGGLE_ADAPTER_REPO",
    f"{KAGGLE_USERNAME}/nemotron-3-nano/transformers/lora-{LORA_STAGE}",
)
KAGGLE_DATASET_REPO = os.environ.get(
    "KAGGLE_DATASET_REPO",
    f"{KAGGLE_USERNAME}/nemotron-{LORA_STAGE}",
)

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "8192"))
DATASET_WORKERS = max(1, int(os.environ.get("DATASET_NUM_PROC", "8")))
DATASET_NUM_PROC = DATASET_WORKERS if DATASET_WORKERS > 1 else None
SEED = int(os.environ.get("SEED", "3407"))

PER_DEVICE_TRAIN_BATCH_SIZE = int(
    os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "2")
)
PER_DEVICE_EVAL_BATCH_SIZE = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("GRADIENT_ACCUMULATION_STEPS", "16")
)
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "10"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "10"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(SAVE_STEPS)))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))

LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
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
    os.environ["WANDB_SILENT"] = "true"
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
    return normalized_prompt + BOXED_ANSWER_INSTRUCTION


def build_conversation(example: dict[str, Any]) -> dict[str, Any]:
    response = clean_text(example.get("response"))
    if response is None:
        raise ValueError("Dataset must be filtered before conversation formatting")

    assistant_message = {
        "role": "assistant",
        "content": response,
        "reasoning_content": clean_text(example.get("reasoning")),
    }

    return {
        "conversations": [
            {
                "role": "user",
                "content": build_user_content(example.get("prompt")),
                "reasoning_content": None,
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
        # template_conversation = [
        #     {
        #         key: value
        #         for key, value in message.items()
        #         if (key != "reasoning_content" and key != "reasoning") or value is not None
        #     }
        #     for message in conversation
        # ]
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
                f"<think>\n{reasoning}\n</think>\n"
                f"{fallback_assistant['content']}"
            )
            fallback_conversation[-1] = fallback_assistant
            text = tokenizer.apply_chat_template(
                # [
                #     {
                #         key: value
                #         for key, value in message.items()
                #         if value is not None
                #     }
                #     for message in fallback_conversation
                # ],
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


def apply_source_options(dataset, split_name: str):
    options = SOURCE_OPTIONS.get(
        split_name,
        {"include": [], "order": [], "exclude": []},
    )
    include_sources = options["include"]
    order_sources = options["order"]
    exclude_sources = set(options["exclude"])
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
        ordered_sources = (
            include_sources
            + [source for source in explicit_order if source in before_wildcard]
            + remaining_sources
            + [source for source in explicit_order if source not in before_wildcard]
        )
    else:
        ordered_sources = include_sources + explicit_order + remaining_sources

    indices_by_source: dict[Any, list[int]] = {}
    for index, source in enumerate(dataset["source"]):
        indices_by_source.setdefault(source, []).append(index)
    selected_indices = [
        index
        for source in ordered_sources
        for index in indices_by_source.get(source, [])
    ]
    ordered = dataset.select(selected_indices, keep_in_memory=KEEP_IN_MEMORY)
    print(
        f"{split_name}: source order={ordered_sources}; "
        f"retained={len(ordered):,}"
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
        is_trainable_example,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: keep non-empty responses",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    if not len(dataset):
        if allow_empty:
            print(f"{split_name}: no trainable examples after filtering")
            return None
        raise ValueError(f"{split_name} has no examples with non-empty responses")

    from datasets import Features, List, Value

    conversation_features = Features(
        {
            "conversations": List(
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "reasoning_content": Value("string"),
                }
            )
        }
    )
    dataset = dataset.map(
        build_conversation,
        remove_columns=dataset.column_names,
        features=conversation_features,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: build conversations",
        keep_in_memory=KEEP_IN_MEMORY,
    )
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
    dataset_dict = load_reasoning_dataset()
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
        # unsloth_tiled_mlp=True,
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        packing=False,
        packing_strategy="bfd",
        max_grad_norm=1e9,
        # remove_unused_columns=True,
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

# %%
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
runtime = trainer_stats.metrics.get("train_runtime", 0.0)
print(f"Training runtime: {runtime:.2f} seconds ({runtime / 60:.2f} minutes)")
print(f"Peak reserved VRAM: {peak_reserved:.2f} GiB")
print(f"Saved LoRA adapter to {ADAPTER_OUTPUT_DIR}")

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
            print(f"Upload to HF failed: {e}, {exc}")
        else:
            print(f"Upload to HF succeeded")
    else:
        print(f"Upload to HF succeeded")

# %%
if PUSH_TO_KAGGLE:
    try:
        import kagglehub
        
        # This creates the model repository or pushes a new version if it already exists
        kagglehub.model_upload(
            handle=KAGGLE_ADAPTER_REPO,
            local_model_dir=ADAPTER_OUTPUT_DIR,
            version_notes=f"LoRA {LORA_STAGE} for unsloth/Nemotron-3-Nano-30B-A3B",
            license_name="Apache 2.0",
        )

        kagglehub.dataset_upload(
            handle=KAGGLE_DATASET_REPO,
            local_dataset_dir=WORKING_DIR,
            version_notes=f"nemotron {LORA_STAGE}",
        )
    except Exception as e:
        print(f"Upload to Kaggle failed: {e}")
    else:
        print(f"Upload to Kaggle succeeded")
