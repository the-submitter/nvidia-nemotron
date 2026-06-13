# %%
from __future__ import annotations

import os
import subprocess
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
# !uv pip install mamba_ssm causal_conv1d --no-index --find-links={wheels_dir}
# !uv pip install --no-deps --upgrade "torchao>=0.16.0" \
#     --no-index --find-links={wheels_dir}
# # !uv pip install vllm --no-index --find-links={wheels_dir}
# # !uv pip install "protobuf<6.0.0" --no-index --find-links={wheels_dir}

# %%
WORKING_DIR = Path(os.environ.get("WORKING_DIR", "/kaggle/working"))
SFT_ADAPTER_PATH = os.environ.get(
    "SFT_ADAPTER_PATH",
    "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/lora-sft/1",
)
BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/default/1",
)
BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/kaggle/input/datasets/rohitraje0493/nemotron-reasoning-dpo",
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

TRAIN_MIN_IDX = optional_nonnegative_int("TRAIN_MIN_IDX")
TRAIN_MAX_IDX = optional_nonnegative_int("TRAIN_MAX_IDX")
EVAL_MIN_IDX = optional_nonnegative_int("EVAL_MIN_IDX")
EVAL_MAX_IDX = optional_nonnegative_int("EVAL_MAX_IDX")

DPO_STAGE = os.environ.get("DPO_STAGE", "dpo")
DPO_VERSION = os.environ.get("DPO_VERSION", "v1")
RUN_NAME = os.environ.get("RUN_NAME", f"nemotron-{DPO_STAGE}-{DPO_VERSION}")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKING_DIR / RUN_NAME)))
ADAPTER_OUTPUT_DIR = Path(
    os.environ.get(
        "ADAPTER_OUTPUT_DIR",
        str(WORKING_DIR / f"nemotron-lora-{DPO_STAGE}-{DPO_VERSION}"),
    )
)
HF_USERNAME = os.environ.get("HF_USERNAME", "the-submitter")
HF_ADAPTER_REPO = os.environ.get(
    "HF_ADAPTER_REPO",
    f"{HF_USERNAME}/nemotron-lora-{DPO_STAGE}-{DPO_VERSION}",
)
KAGGLE_ADAPTER_REPO = os.environ.get(
    "KAGGLE_ADAPTER_REPO",
    f"{KAGGLE_USERNAME}/nemotron-3-nano/transformers/lora-{DPO_STAGE}",
)
KAGGLE_DATASET_REPO = os.environ.get(
    "KAGGLE_DATASET_REPO",
    f"{KAGGLE_USERNAME}/nemotron-{DPO_STAGE}",
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
    os.environ.get("GRADIENT_ACCUMULATION_STEPS", "16")
)
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "10"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "50"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(SAVE_STEPS)))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))

DPO_BETA = float(os.environ.get("DPO_BETA", "0.1"))
DPO_LOSS_TYPE = os.environ.get("DPO_LOSS_TYPE", "sigmoid")
PRECOMPUTE_REF_LOG_PROBS = os.environ.get(
    "PRECOMPUTE_REF_LOG_PROBS",
    "1",
).lower() in {"1", "true", "yes"}
REPORT_TO = os.environ.get("REPORT_TO", "wandb")
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT")
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


def build_user_content(prompt: Any) -> str:
    normalized_prompt = clean_text(prompt)
    if normalized_prompt is None:
        raise ValueError("Cannot format DPO data without a prompt")
    return normalized_prompt + BOXED_ANSWER_INSTRUCTION


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
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


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


def prepare_split(dataset, tokenizer, split_name: str):
    original_size = len(dataset)
    dataset = dataset.filter(
        is_preference_example,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: keep complete preferences",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    if not len(dataset):
        raise ValueError(
            f"{split_name} has no examples with non-empty chosen and rejected"
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
    dataset_dict = load_preference_dataset()
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
            is_preference_example,
            num_proc=DATASET_NUM_PROC,
            desc=f"{EVAL_SPLIT}: check complete preferences",
            keep_in_memory=KEEP_IN_MEMORY,
        )
        if len(filtered_eval):
            eval_dataset = filtered_eval.map(
                render_dpo_example,
                fn_kwargs={"tokenizer": tokenizer},
                remove_columns=filtered_eval.column_names,
                num_proc=DATASET_NUM_PROC,
                desc=f"{EVAL_SPLIT}: apply DPO chat template",
                keep_in_memory=KEEP_IN_MEMORY,
            )
            eval_dataset = select_index_range(
                eval_dataset,
                EVAL_MIN_IDX,
                EVAL_MAX_IDX,
                EVAL_SPLIT,
            )
            print(
                f"{EVAL_SPLIT}: retained {len(eval_dataset):,}/"
                f"{len(dataset_dict[EVAL_SPLIT]):,} examples"
            )
    return train_dataset, eval_dataset


# %%
def load_model_and_tokenizer():
    from unsloth import FastLanguageModel

    adapter_path = Path(SFT_ADAPTER_PATH)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"SFT LoRA adapter_config.json does not exist under {adapter_path}"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
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

from unsloth import FastLanguageModel, PatchDPOTrainer  # noqa: F401

PatchDPOTrainer()

import torch
from trl import DPOConfig, DPOTrainer

if not torch.cuda.is_available():
    raise RuntimeError("DPO training requires a CUDA GPU")

# %%
model, tokenizer = load_model_and_tokenizer()

# %%
train_dataset, eval_dataset = prepare_datasets(tokenizer)

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
        beta=DPO_BETA,
        loss_type=[DPO_LOSS_TYPE],
        precompute_ref_log_probs=PRECOMPUTE_REF_LOG_PROBS,
        max_grad_norm=1e9,
    ),
)

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

# %%
trainer_stats = dpo_trainer.train(
    resume_from_checkpoint=resolve_resume_from_checkpoint()
)

# %%
dpo_trainer.save_state()
model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))

# %%
subprocess.run(f'sed -i "s|{BASE_MODEL_PATH}|{BASE_MODEL_ID}|g" {ADAPTER_OUTPUT_DIR}/README.md', shell=True)
subprocess.run(f'sed -i "s|{BASE_MODEL_PATH}|{BASE_MODEL_ID}|g" {ADAPTER_OUTPUT_DIR}/adapter_config.json', shell=True)

# %%
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
runtime = trainer_stats.metrics.get("train_runtime", 0.0)
print(f"Training runtime: {runtime:.2f} seconds ({runtime / 60:.2f} minutes)")
print(f"Peak reserved VRAM: {peak_reserved:.2f} GiB")
print(f"Saved DPO-trained LoRA adapter to {ADAPTER_OUTPUT_DIR}")

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
            version_notes=f"Nemotron LoRA continued from sft to {DPO_STAGE}",
            license_name="Apache 2.0",
        )

        kagglehub.dataset_upload(
            handle=KAGGLE_DATASET_REPO,
            local_dataset_dir=WORKING_DIR,
            version_notes=f"nemotron {DPO_STAGE}",
        )
    except Exception as error:
        print(f"Upload to Kaggle failed: {error}")
    else:
        print("Upload to Kaggle succeeded")
