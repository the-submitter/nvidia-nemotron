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
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", 8192)) # 8192 # 32768
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "False").lower() == "true"

DATASET_NAME = os.environ.get("DATASET_NAME", "the-submitter/code-62k")

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

if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "GenBot-train-code"
if "WANDB_NAME" not in os.environ:
    os.environ["WANDB_NAME"] = f'{MODEL_NAME.split("/")[-1]}-sft'

SAVE_PREFIX = os.environ.get("SAVE_PREFIX", "genbot-code-e4b")
HF_UPLOAD_USERNAME = os.environ.get("HF_UPLOAD_USERNAME", "the-submitter")

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", os.cpu_count()))
DOMAIN = os.environ.get("DOMAIN", "code")

print(
    f"MODEL_NAME: {MODEL_NAME}\nMAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}\nENABLE_THINKING: {ENABLE_THINKING}"
    f"\nDATASET_NAME: {DATASET_NAME}"
    f"\nTRAINER_MAX_STEPS: {TRAINER_MAX_STEPS}\nTRAINER_SAVE_STEPS: {TRAINER_SAVE_STEPS}"
    f"\nTRAINER_MAX_ATTEMPTS: {TRAINER_MAX_ATTEMPTS}\nTRAINER_BACKOFF_FACTOR: {TRAINER_BACKOFF_FACTOR}"
    f"\nTRAINER_ENABLE_EVAL: {TRAINER_ENABLE_EVAL}\nTRAINER_EVAL_STEPS: {TRAINER_EVAL_STEPS}"
    f"\nTRAINER_EARLY_STOP_PATIENCE: {TRAINER_EARLY_STOP_PATIENCE}"
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
#
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# %%
from unsloth import FastModel
import torch

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, processor = FastModel.from_pretrained(
    model_name = MODEL_NAME,
    dtype = None, # None for auto detection
    max_seq_length = MAX_SEQ_LENGTH, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False,
    load_in_16bit = False,
    # float8_kv_cache=True,
    full_finetuning = False, # [NEW!] We have full finetuning now!
    token = HF_KEY, # HF Token for gated models
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
# # Gemma 4 can process Text, Vision and Audio!
#
# Let's first experience how Gemma 4 can handle multimodal inputs. We use Gemma 4's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`

# %%
from transformers import TextStreamer
# Helper function for inference
def do_gemma_4_inference(model, processor, messages, max_new_tokens = 128):
    _ = model.generate(
        **processor.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
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
# # Gemma 4 can see images!
#
# <img src="https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg" alt="Alt text" height="256">

# %%
if False:
    sloth_link = "https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg"
    
    messages = [{
        "role" : "user",
        "content": [
            { "type": "image", "image" : sloth_link },
            { "type": "text",  "text" : "Which films does this animal feature in?" }
        ]
    }]
    # You might have to wait 1 minute for Unsloth's auto compiler
    do_gemma_4_inference(model, processor, messages, max_new_tokens = 256)

# %% [markdown]
# Let's make a poem about sloths!

# %%
if False:
    messages = [{
        "role": "user",
        "content": [{ "type" : "text",
                      "text" : "Write a poem about sloths." }]
    }]
    do_gemma_4_inference(model, processor, messages)

# %% [markdown]
# # Let's finetune Gemma 4!
#
# You can finetune the vision and text parts for now through selection - the audio part can also be finetuned - we're working to make it selectable as well!

# %% [markdown]
# To add new tokens to model vocabulary:

# %%
if False:
    # Define your quantized values or special tokens
    new_tokens = ["<q_val_0>", "<q_val_1>", "<q_val_2>"] 

    # Add to tokenizer
    num_added_toks = processor.tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_toks} tokens")

    # Resize both input embeddings and output head (lm_head)
    model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=8)

    # Update configuration
    model.config.vocab_size = len(processor.tokenizer)

# %% [markdown]
# We now add LoRA adapters so we only need to update a small amount of parameters!

# %%
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora=False,
    loftq_config=None,
    # target_modules="all-linear",
    # modules_to_save=["lm_head", "embed_tokens"], # save the lm_head and embed_tokens to train the special tokens
)

# %% [markdown]
# <a name="Data"></a>
# ### Data Prep
# We now use the `Gemma-4` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-4 renders multi turn conversations like below:
#
# ```
# <bos><|turn>user
# Hello<turn|>
# <|turn>model
# Hey there!<turn|>
# ```
# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3, gemma-4` and more.

# %%
# from unsloth.chat_templates import get_chat_template
# processor = get_chat_template(
#     processor,
#     chat_template = "gemma-4-thinking",
# )

# %%
# # patch to enable "reasoning" even when tool_calls is empty
# target = "and message.get('tool_calls')"
# processor.chat_template = processor.chat_template.replace(target, "")

# from functools import partial, update_wrapper
# processor.apply_chat_template = partial(processor.apply_chat_template, enable_thinking=True)
# update_wrapper(processor.apply_chat_template, processor.apply_chat_template.func)

# %% [markdown]
# We get the first 3000 rows of the dataset

# %%
from datasets import load_dataset

dataset = load_dataset(DATASET_NAME, token=HF_KEY, num_proc=MAX_WORKERS)

# %% [markdown]
# We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!

# %%
# from unsloth.chat_templates import standardize_data_formats
# dataset = standardize_data_formats(dataset)

# %% [markdown]
# Let's see how row 100 looks like!

# %%
dataset["train"][0]

# %% [markdown]
# To format the dataset, all vision fine-tuning tasks should follow this format:
#
# ```python
# [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": instruction},
#             {"type": "image", "image": sample["image"]},
#         ],
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": instruction},
#             {"type": "image", "image": sample["image"]},
#         ],
#     },
# ]
# ```

# %%
# def convert_to_conversation(sample):
#     conversation = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": sample["prompt"]},
#             ],
#         },
#         {"role": "assistant", "content": [{"type": "text", "text": sample["response"]}]},
#     ]
#     return {"messages": conversation}

# %%
import sys
import importlib

if str(repo_dir) not in sys.path:
    sys.path.append(str(repo_dir))
utils = importlib.import_module(f"train.domains.{DOMAIN}.utils")
convert_to_conversation = getattr(utils, "convert_to_conversation", lambda sample: sample)

def convert_dataset(dataset, convert_fn):
    final_dataset = []
    for sample in dataset:
        conv = convert_fn(sample)
        if isinstance(conv, dict):
            final_dataset.append(conv)
        else:
            final_dataset.extend(conv)
    return final_dataset

dataset_train = convert_dataset(dataset["train"], convert_to_conversation)
dataset_validation = None
try:
    if TRAINER_ENABLE_EVAL:
        dataset_validation = convert_dataset(dataset["validation"].take(10), convert_to_conversation)
except Exception:
    pass
try:
    dataset_test = convert_dataset(dataset["test"].take(10), convert_to_conversation)
except Exception:
    dataset_test = None
del dataset

# %%
dataset_train[0]

# %% [markdown]
# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# %%
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
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
    trainer = SFTTrainer(
        model = model,
        # tokenizer = tokenizer,
        train_dataset = dataset_train,
        eval_dataset = dataset_validation, # Can set up evaluation!
        processing_class=processor.tokenizer,
        data_collator = UnslothVisionDataCollator(
            model, 
            processor,
            train_on_responses_only=True,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
        ),
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 10,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = max_steps,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
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
            report_to = "wandb", # Use TrackIO/WandB etc
            output_dir = output_dir,
            padding_free=True,
            # max_grad_norm=0.5,

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = max_length, # 2048,
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

# %% [markdown]
# # Let's train the model!
#
# To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

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
# <a name="Inference"></a>
# ### Inference
# Let's run the model via Unsloth native inference! According to the `Gemma-4` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`

# %%
if True:
    do_gemma_4_inference(model, processor, dataset_test[0]["messages"][:-1])

# %% [markdown]
# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
#
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# %%
model.save_pretrained(f"{SAVE_PREFIX}-lora")  # Local saving
processor.save_pretrained(f"{SAVE_PREFIX}-lora")
model.push_to_hub(f"{HF_UPLOAD_USERNAME}/{SAVE_PREFIX}-lora", token = HF_KEY, private=True) # Online saving
processor.push_to_hub(f"{HF_UPLOAD_USERNAME}/{SAVE_PREFIX}-lora", token = HF_KEY, private=True) # Online saving

# %% [markdown]
# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# %%
if False:
    from unsloth import FastModel

    try:
        import torch
        import gc
        # free the memory again
        del model
        del processor
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass

    model, processor = FastModel.from_pretrained(
        model_name = f"{SAVE_PREFIX}-lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
        # device_map="balanced",
        dtype = None, # None for auto detection
        unsloth_tiled_mlp=True,
        gpu_memory_utilization=0.95,
    )

    do_gemma_4_inference(model, processor, dataset_test[0]["messages"][:-1])

# %% [markdown]
# ### Saving to float16 for VLLM
#
# We also support saving to `float16` directly for deployment! We save it in the folder `gemma-4-finetune`. Set `if False` to `if True` to let it run!

# %%
if False: # Change to True to save finetune!
    model.save_pretrained_merged(f"/tmp/{SAVE_PREFIX}", processor)

# %% [markdown]
# If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# %%
if True: # Change to True to upload finetune
    original_path = os.getcwd()
    os.chdir("/tmp")    # prevent Kaggle working dir out of disk error
    model_slug = f"{HF_UPLOAD_USERNAME}/{SAVE_PREFIX}"
    try:
        model.push_to_hub_merged(
            model_slug,
            processor,
            token = HF_KEY,
            private=True,
        )
    finally:
        # import shutil
        # shutil.rmtree(f"/tmp/{model_slug}", ignore_errors=True)
        os.chdir(original_path)

# %% [markdown]
# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!

# %%
if False: # Change to True to save to GGUF
    methods = ["q4_k_m", "q3_k_m", "q2_k"]  # "iq2_xxs"
    model.save_pretrained_gguf(
        f"/tmp/{SAVE_PREFIX}-gguf",
        processor,
        quantization_method = methods,
    )

# %% [markdown]
# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# %%
if False: # Change to True to upload GGUF
    methods = ["q4_k_m", "q3_k_m", "q2_k"]  # "iq2_xxs"
    original_path = os.getcwd()    # prevent Kaggle working dir out of disk error
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
        # import shutil
        # shutil.rmtree(f"/tmp/{model_slug}", ignore_errors=True)
        os.chdir(original_path)

# %% [markdown]
# Now, use the `gemma-4-finetune.gguf` file or `gemma-4-finetune-Q4_K_M.gguf` file in llama.cpp.
#
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
#
# Some other resources:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!
#
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
#
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
#
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
