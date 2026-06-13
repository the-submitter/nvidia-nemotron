# %%
from __future__ import annotations
import os

# User secrets
try:
    from kaggle_secrets import UserSecretsClient    # type: ignore

    user_secrets = UserSecretsClient()
    GITPAT = user_secrets.get_secret("GITPAT")
    KAGGLE_KEY = user_secrets.get_secret("KAGGLE_KEY")
    KAGGLE_USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
    HF_KEY = user_secrets.get_secret("HF_KEY")
    WANDB_KEY = user_secrets.get_secret("WANDB_KEY")

    if KAGGLE_KEY:
        os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    if KAGGLE_USERNAME:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    if HF_KEY:
        os.environ["HF_TOKEN"] = HF_KEY
    if WANDB_KEY:
        os.environ["WANDB_API_KEY"] = WANDB_KEY

except Exception:
    GITPAT = os.environ.get("GITPAT")
    KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
    KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
    HF_KEY = os.environ.get("HF_KEY")
    if HF_KEY and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = HF_KEY
    WANDB_KEY = os.environ.get("WANDB_KEY")
    if WANDB_KEY and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = WANDB_KEY

# %%
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/gemma-4-E4B-it")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", 8192))
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "False").lower() == "true"

DATASET_NAME = os.environ.get("DATASET_NAME", "the-submitter/code-62k-preference")

TRAINER_MAX_STEPS = int(os.environ.get("TRAINER_MAX_STEPS", 1000))
TRAINER_SAVE_STEPS = int(os.environ.get("TRAINER_SAVE_STEPS", 50))
TRAINER_MAX_ATTEMPTS = int(os.environ.get("TRAINER_MAX_ATTEMPTS", 1))
TRAINER_BACKOFF_FACTOR = float(os.environ.get("TRAINER_BACKOFF_FACTOR", 0.75))
TRAINER_ENABLE_EVAL = os.environ.get("TRAINER_ENABLE_EVAL", "True").lower() == "true"
TRAINER_EVAL_STEPS = int(os.environ.get("TRAINER_EVAL_STEPS", 50))
try:
    TRAINER_EARLY_STOP_PATIENCE = int(os.environ.get("TRAINER_EARLY_STOP_PATIENCE", 3))
except Exception:
    TRAINER_EARLY_STOP_PATIENCE = None
TRAINER_OUTPUT_DIR = os.environ.get("TRAINER_OUTPUT_DIR", "outputs")

DPO_BETA = float(os.environ.get("DPO_BETA", 0.1))
DPO_LOSS_TYPE = os.environ.get("DPO_LOSS_TYPE", "sigmoid")
DPO_MAX_PROMPT_LENGTH_FACTOR = os.environ.get("DPO_MAX_PROMPT_LENGTH_FACTOR", 0.5)
if DPO_MAX_PROMPT_LENGTH_FACTOR is not None:
    try:
        DPO_MAX_PROMPT_LENGTH_FACTOR = max(0.0, min(1.0, float(DPO_MAX_PROMPT_LENGTH_FACTOR)))
    except Exception:
        DPO_MAX_PROMPT_LENGTH_FACTOR = None
DPO_MAX_COMPLETION_LENGTH_FACTOR = os.environ.get("DPO_MAX_COMPLETION_LENGTH_FACTOR")
if DPO_MAX_COMPLETION_LENGTH_FACTOR is not None:
    try:
        DPO_MAX_COMPLETION_LENGTH_FACTOR = max(0.0, min(1.0, float(DPO_MAX_COMPLETION_LENGTH_FACTOR)))
    except Exception:
        DPO_MAX_COMPLETION_LENGTH_FACTOR = None

if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "GenBot-train-code"
if "WANDB_NAME" not in os.environ:
    os.environ["WANDB_NAME"] = f'{MODEL_NAME.split("/")[-1]}-dpo'

SAVE_PREFIX = os.environ.get("SAVE_PREFIX", "genbot-code-e4b")
HF_UPLOAD_USERNAME = os.environ.get("HF_UPLOAD_USERNAME", "the-submitter")

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", os.cpu_count() or 1))
DOMAIN = os.environ.get("DOMAIN", "code")

print(
    f"MODEL_NAME: {MODEL_NAME}\nMAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}\nENABLE_THINKING: {ENABLE_THINKING}"
    f"\nDATASET_NAME: {DATASET_NAME}"
    f"\nTRAINER_MAX_STEPS: {TRAINER_MAX_STEPS}\nTRAINER_SAVE_STEPS: {TRAINER_SAVE_STEPS}"
    f"\nTRAINER_MAX_ATTEMPTS: {TRAINER_MAX_ATTEMPTS}\nTRAINER_BACKOFF_FACTOR: {TRAINER_BACKOFF_FACTOR}"
    f"\nTRAINER_ENABLE_EVAL: {TRAINER_ENABLE_EVAL}\nTRAINER_EVAL_STEPS: {TRAINER_EVAL_STEPS}"
    f"\nTRAINER_EARLY_STOP_PATIENCE: {TRAINER_EARLY_STOP_PATIENCE}"
    f"\nDPO_BETA: {DPO_BETA}\nDPO_LOSS_TYPE: {DPO_LOSS_TYPE}"
    f"\nDPO_MAX_PROMPT_LENGTH: {DPO_MAX_PROMPT_LENGTH_FACTOR}\nDPO_MAX_COMPLETION_LENGTH: {DPO_MAX_COMPLETION_LENGTH_FACTOR}"
    f"\nSAVE_PREFIX: {SAVE_PREFIX}\nHF_UPLOAD_USERNAME: {HF_UPLOAD_USERNAME}"
    f"\nMAX_WORKERS: {MAX_WORKERS}\nDOMAIN: {DOMAIN}"
)

# %%
from pathlib import Path

repo_name = "genbot"
working_dir = Path("/kaggle/working")
repo_dir = working_dir / repo_name
repo_url = f"https://{GITPAT}@github.com/the-submitter/{repo_name}.git"

# %%
# if os.path.exists(repo_dir / ".git"):
#     print("pull")
#     %cd {repo_dir}
# #     !git config pull.rebase true
# #     !git fetch origin
# #     !git stash
# #     !git pull
# #     !git checkout main
# else:
#     print("clone")
#     %cd {repo_dir.parent}
# #     !git clone -b main {repo_url}
#     %cd {repo_dir}

# %% [markdown]
# ### Installation

# %%
# %%capture
# !uv pip install -qqq -r train/requirements/main.txt --torch-backend auto
# !uv pip install -qqq -r train/domains/{DOMAIN}/requirements.txt --torch-backend auto

# %% [markdown]
# ### Unsloth

# %%
from unsloth import FastModel
import torch

gemma4_models = [
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
]

model, processor = FastModel.from_pretrained(
    model_name = MODEL_NAME,
    dtype = None,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
    load_in_8bit = False,
    load_in_16bit = False,
    # float8_kv_cache=True,
    full_finetuning = False,
    token = HF_KEY,
    # device_map = "balanced", # Use 2x Tesla T4s on Kaggle
    use_gradient_checkpointing="unsloth",
    unsloth_tiled_mlp=True,
    gpu_memory_utilization=0.95,
)

# %%
# patch to enable "reasoning" even when tool_calls is empty
target = "and message.get('tool_calls')"
processor.chat_template = processor.chat_template.replace(target, "")

if ENABLE_THINKING:
    from functools import partial, update_wrapper
    processor.apply_chat_template = partial(processor.apply_chat_template, enable_thinking=True)
    update_wrapper(processor.apply_chat_template, processor.apply_chat_template.func)

# %% [markdown]
# ### Inference helper

# %%
from transformers import TextStreamer


def do_gemma_4_inference(model, processor, messages, max_new_tokens = 128):
    _ = model.generate(
        **processor.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to(model.device),
        max_new_tokens = max_new_tokens,
        use_cache = True,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(processor, skip_prompt = True),
    )


# %% [markdown]
# ### LoRA

# %%
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora=False,
    loftq_config=None,
)

# %% [markdown]
# ### Data Prep

# %%
from datasets import load_dataset

dataset = load_dataset(DATASET_NAME, token=HF_KEY, num_proc=MAX_WORKERS)

# %%
import sys
import importlib

if str(repo_dir) not in sys.path:
    sys.path.append(str(repo_dir))
utils = importlib.import_module(f"train.domains.{DOMAIN}.utils")
convert_to_dpo = getattr(utils, "convert_to_dpo", lambda sample: sample)

def convert_dataset(dataset, convert_fn):
    final_dataset = []
    for sample in dataset:
        conv = convert_fn(sample)
        if isinstance(conv, dict):
            final_dataset.append(conv)
        else:
            final_dataset.extend(conv)
    return final_dataset

dataset_train = convert_dataset(dataset["train"], convert_to_dpo)
dataset_validation = None
try:
    if TRAINER_ENABLE_EVAL:
        dataset_validation = convert_dataset(dataset["validation"].take(10), convert_to_dpo)
except Exception:
    pass
try:
    dataset_test = convert_dataset(dataset["test"].take(10), convert_to_dpo)
except Exception:
    dataset_test = None
del dataset

# %%
dataset_train[0]

# %% [markdown]
# ### Train the model

# %%
# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()

from unsloth.trainer import UnslothVisionDataCollator
from trl import DPOTrainer, DPOConfig
from transformers import EarlyStoppingCallback


def get_trainer(
        max_length: int = MAX_SEQ_LENGTH,
        max_steps: int = TRAINER_MAX_STEPS,
        save_steps: int = TRAINER_SAVE_STEPS,
        dataset_validation: any | None = dataset_validation,
        eval_steps: int = TRAINER_EVAL_STEPS,
        early_stop_patience: int | None = TRAINER_EARLY_STOP_PATIENCE,
        output_dir: str = TRAINER_OUTPUT_DIR,
):
    trainer = DPOTrainer(
        model = model,
        ref_model = None,
        train_dataset = dataset_train,
        eval_dataset = dataset_validation,
        processing_class = processor.tokenizer,
        data_collator = UnslothVisionDataCollator(
            model, 
            processor,
            train_on_responses_only=True,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
        ),
        args = DPOConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = max_steps,
            learning_rate = 2e-4,
            logging_steps = 1,
            save_strategy = "steps",
            save_steps = save_steps,
            per_device_eval_batch_size = 1,
            eval_do_concat_batches = False,
            eval_strategy = "steps" if dataset_validation else "no",
            eval_steps = eval_steps if dataset_validation else None,
            eval_accumulation_steps = 4 if dataset_validation else None,
            save_total_limit = 2,
            load_best_model_at_end = bool(dataset_validation),
            metric_for_best_model = "eval_loss" if dataset_validation else "loss",
            greater_is_better = False,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "cosine",
            seed = 3407,
            report_to = "wandb",
            output_dir = output_dir,
            padding_free=True,

            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = max_length,
            max_prompt_length = (
                DPO_MAX_PROMPT_LENGTH_FACTOR * max_length 
                if DPO_MAX_PROMPT_LENGTH_FACTOR is not None 
                else None
            ),
            max_completion_length = (
                DPO_MAX_COMPLETION_LENGTH_FACTOR * max_length 
                if DPO_MAX_COMPLETION_LENGTH_FACTOR is not None 
                else None
            ),
            beta = DPO_BETA,
            loss_type = DPO_LOSS_TYPE,
            precompute_ref_log_probs=True,
        ),
    )

    if early_stop_patience and dataset_validation:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience = early_stop_patience))

    return trainer


trainer = get_trainer()

# %%
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %%
new_max_length = max_length = MAX_SEQ_LENGTH
for attempt in range(TRAINER_MAX_ATTEMPTS):
    try:
        try:
            trainer_stats = trainer.train(resume_from_checkpoint=True)
        except ValueError as e:
            if "No valid checkpoint found" in str(e):
                print("No valid checkpoints found. Performing fresh train...")
                trainer_stats = trainer.train()
        break

    except torch.OutOfMemoryError:
        print(f"OOM detected! Attempting recovery ({attempt + 1}/{TRAINER_MAX_ATTEMPTS})...")
        print(f"Actual trainer.args.max_length: {trainer.args.max_length}")
        del trainer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if attempt < TRAINER_MAX_ATTEMPTS - 1:
            max_length = new_max_length
            new_max_length = int(max_length * TRAINER_BACKOFF_FACTOR)
            trainer = get_trainer(max_length=new_max_length)
            print(f"Reduced max_length from {max_length} to {new_max_length}.")
else:
    print("Ran out of retry attempts.")
    import sys
    sys.exit(137)
    # raise RuntimeError("Ran out of retry attempts.")

# %%
# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# %% [markdown]
# ### Inference

# %%
if dataset_test:
    do_gemma_4_inference(model, processor, dataset_test[0]["prompt"])

# %% [markdown]
# ### Saving, loading finetuned models

# %%
model.save_pretrained(f"{SAVE_PREFIX}-lora")
processor.save_pretrained(f"{SAVE_PREFIX}-lora")
model.push_to_hub(f"{HF_UPLOAD_USERNAME}/{SAVE_PREFIX}-lora", token = HF_KEY, private=True)
processor.push_to_hub(f"{HF_UPLOAD_USERNAME}/{SAVE_PREFIX}-lora", token = HF_KEY, private=True)

# %%
if False:
    from unsloth import FastModel

    try:
        import gc
        del model
        del processor
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass

    model, processor = FastModel.from_pretrained(
        model_name = f"{SAVE_PREFIX}-lora",
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
        dtype = None,
        unsloth_tiled_mlp=True,
        gpu_memory_utilization=0.95,
    )

    if dataset_test:
        do_gemma_4_inference(model, processor, dataset_test[0]["prompt"])

# %%
if False:
    model.save_pretrained_merged(f"/tmp/{SAVE_PREFIX}", processor)

# %%
if True:
    original_path = os.getcwd()
    os.chdir("/tmp")
    model_slug = f"{HF_UPLOAD_USERNAME}/{SAVE_PREFIX}"
    try:
        model.push_to_hub_merged(
            model_slug,
            processor,
            token = HF_KEY,
            private=True,
        )
    finally:
        os.chdir(original_path)

# %%
if False:
    methods = ["q4_k_m", "q3_k_m", "q2_k", "iq2_xxs"]
    model.save_pretrained_gguf(
        f"/tmp/{SAVE_PREFIX}-gguf",
        processor,
        quantization_method = methods,
    )

# %%
if False:
    methods = ["q4_k_m", "q3_k_m", "q2_k", "iq2_xxs"]
    original_path = os.getcwd()
    os.chdir("/tmp")
    model_slug = f"{HF_UPLOAD_USERNAME}/{SAVE_PREFIX}-gguf"
    try:
        model.push_to_hub_gguf(
            model_slug,
            processor,
            quantization_method = methods,
            token = HF_KEY,
            private=True,
        )
    finally:
        os.chdir(original_path)
