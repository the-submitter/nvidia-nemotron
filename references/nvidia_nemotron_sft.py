# %% [markdown]
# To run this, press "*Runtime*" and press "*Run all*" on your A100 Google Colab Pro instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
#
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
#
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it

# %% [markdown]
# ### News

# %% [markdown]
# Introducing **Unsloth Studio** - a new open source, no-code web UI to train and run LLMs. [Blog](https://unsloth.ai/docs/new/studio) • [Notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb)
#
# <table><tr>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FxV1PO5DbF3ksB51nE2Tw%252Fmore%2520cropped%2520ui%2520for%2520homepage.png%3Falt%3Dmedia%26token%3Df75942c9-3d8d-4b59-8ba2-1a4a38de1b86&width=376&dpr=3&quality=100&sign=a663c397&sv=2" width="200" height="120" alt="Unsloth Studio Training UI"></a><br><sub><b>Train models</b> — no code needed</sub></td>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FRCnTAZ6Uh88DIlU3g0Ij%252Fmainpage%2520unsloth.png%3Falt%3Dmedia%26token%3D837c96b6-bd09-4e81-bc76-fa50421e9bfb&width=376&dpr=3&quality=100&sign=c1a39da1&sv=2" width="200" height="120" alt="Unsloth Studio Chat UI"></a><br><sub><b>Run GGUF models</b> on Mac, Windows & Linux</sub></td>
# </tr></table>
#
# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
#
# Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)
#
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
#
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).

# %% [markdown]
# ### Installation

# %%
# # # %%capture
# import os, importlib.util
# # # !pip install --upgrade -qqq uv
# if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
#     try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
#     except: _numpy = "numpy"; _pil = "pillow"
#     # !uv pip install -qqq \
# #         "torch==2.7.1" "triton>=3.3.0" {_numpy} {_pil} torchvision bitsandbytes "transformers==4.56.2" \
# #         "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
# #         "unsloth[base] @ git+https://github.com/unslothai/unsloth"
#     # !uv pip install -qqq --no-deps "torchcodec==0.5"
# elif importlib.util.find_spec("unsloth") is None:
#     pass
#     # !uv pip install -qqq unsloth
# # # !uv pip install --upgrade --no-deps transformers==4.56.2 "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo

# # Mamba is supported only on torch==2.7.1. If you have newer torch versions, please wait 30 minutes!
# # # !uv pip install --no-build-isolation mamba_ssm==2.2.5 causal_conv1d==1.5.2
# # # !uv pip install --no-deps --upgrade "torchao>=0.16.0"

# %%
from __future__ import annotations
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
# import os

# # User secrets
# try:
#     from kaggle_secrets import UserSecretsClient    # type: ignore

#     user_secrets = UserSecretsClient()
#     GITPAT = user_secrets.get_secret("GITPAT")
#     KAGGLE_KEY = user_secrets.get_secret("KAGGLE_KEY")
#     KAGGLE_USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
#     HF_KEY = user_secrets.get_secret("HF_KEY")
#     WANDB_KEY = user_secrets.get_secret("WANDB_KEY")

#     if KAGGLE_KEY:
#         os.environ["KAGGLE_KEY"] = KAGGLE_KEY
#     if KAGGLE_USERNAME:
#         os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
#     if HF_KEY:
#         os.environ["HF_TOKEN"] = HF_KEY
#     if WANDB_KEY:
#         os.environ["WANDB_API_KEY"] = WANDB_KEY

# except Exception:
#     GITPAT = os.environ.get("GITPAT")
#     KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
#     KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
#     HF_KEY = os.environ.get("HF_KEY")
#     if HF_KEY and not os.environ.get("HF_TOKEN"):
#         os.environ["HF_TOKEN"] = HF_KEY
#     WANDB_KEY = os.environ.get("WANDB_KEY")
#     if WANDB_KEY and not os.environ.get("WANDB_API_KEY"):
#         os.environ["WANDB_API_KEY"] = WANDB_KEY
# HF_USERNAME = "the-submitter"

# %% [markdown]
# ### Unsloth

# %%
from unsloth import FastLanguageModel
import torch

fourbit_models = [
    "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", # Qwen 14B 2x faster
    "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",

    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
] # More models at https://huggingface.co/unsloth
model_dir = "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/default/1"

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/Nemotron-3-Nano-30B-A3B",
    model_name = model_dir,
    max_seq_length = 7680, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    trust_remote_code = True,
    unsloth_force_compile = True,
    attn_implementation = "eager",
    token = HF_KEY, # HF Token for gated models
    use_gradient_checkpointing="unsloth",
    unsloth_tiled_mlp=True,
    gpu_memory_utilization=0.95,
)

# %% [markdown]
# We now add LoRA adapters so we only need to update a small amount of parameters!

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "in_proj", "out_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# %% [markdown]
# <a name="Data"></a>
# ### Data Prep
# We now use the `Nemotron` format for conversation style finetunes. We use the [Open Math Reasoning](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) dataset which was used to win the [AIMO](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/leaderboard) (AI Mathematical Olympiad - Progress Prize 2) challenge! We sample 10% of verifiable reasoning traces that used DeepSeek R1, and which got > 95% accuracy. Nemotron renders multi turn conversations like below:
#
# ```
# <|im_start|>user
# Hello!<|im_end|>
# <|im_start|>assistant
# Hey there!<|im_end|>
# ```

# %%
from datasets import load_dataset
dataset_dir = "/kaggle/input/datasets/rohitraje0493/nemotron-reasoning"
dataset = load_dataset("parquet", data_dir=dataset_dir)

# %% [markdown]
# We now convert the reasoning dataset into conversational format:

# %%
def generate_conversation(examples):
    prompts  = examples["prompt"]
    responses = examples["response"]
    reasonings = examples["reasoning"]
    conversations = []
    for prompt, response, reasoning in zip(prompts, responses, reasonings):
        if response:
            user_content = prompt + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
            conversations.append([
                {"role" : "user",      "content" : user_content},
                {"role" : "assistant", "content" : response, "reasoning_content": reasoning},
            ])
    return { "conversations": conversations, }

for ds_split in dataset:
    dataset[ds_split] = dataset[ds_split].map(generate_conversation, batched = True)

# %% [markdown]
# We now have to apply the chat template for `Nemotron` onto the conversations, and save it to `text`.

# %%
def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# %% [markdown]
# Let's see how the chat template did!

# %%
dataset[100]['text']

# %% [markdown]
# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# %%
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        save_strategy = "steps",
        save_steps = 50,
        per_device_eval_batch_size = 1,
        eval_do_concat_batches = False,
        eval_strategy = "steps", # "no"
        eval_steps = 50,    # None
        eval_accumulation_steps = 4,    # None
        save_total_limit = 2,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",    # "loss"
        greater_is_better = False,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
        output_dir = "output",
        padding_free=True,
        packing=True,
        packing_strategy="bfd", # "wrapped"
        # max_grad_norm=0.5,
        # assistant_only_loss=True,
        # completion_only_loss=True,

        # # You MUST put the below items for vision finetuning:
        # remove_unused_columns = False,
        # dataset_text_field = "",
        # dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 7680, # 2048,
    ),
)

# %% [markdown]
# We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!

# %%
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# %% [markdown]
# Let's verify masking the instruction part is done! Let's print the 100th row again.

# %%
tokenizer.decode(trainer.train_dataset[100]["input_ids"])

# %% [markdown]
# Now let's print the masked out example - you should see only the answer is present:

# %%
tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# %%
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %% [markdown]
# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# %%
trainer_stats = trainer.train()

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
# Let's run the model via Unsloth native inference!

# %%
messages = [
    {"role" : "user", "content" : "Continue the sequence: 1, 1, 2, 3, 5, 8,"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 1000, # Increase for longer outputs!
    temperature = 0.7, top_p = 0.8, top_k = 20,
    use_cache = False,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# %% [markdown]
# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
#
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# %%
model.save_pretrained("nemotron_lora")  # Local saving
tokenizer.save_pretrained("nemotron_lora")
# model.push_to_hub(f"{HF_USERNAME}/nemotron-lora", token = HF_KEY, private=True) # Online saving
# tokenizer.push_to_hub(f"{HF_USERNAME}/nemotron-lora", token = HF_KEY, private=True) # Online saving

# %% [markdown]
# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# %%
if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "nemotron_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

# %% [markdown]
# ### Saving to float16 for VLLM
#
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# %%
# Merge to 16bit
if False:
    model.save_pretrained_merged("nemotron_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged(f"{HF_USERNAME}/nemotron-finetune-16bit", tokenizer, save_method = "merged_16bit", token = HF_KEY, private=True)

# Merge to 4bit
if False:
    model.save_pretrained_merged("nemotron_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged(f"{HF_USERNAME}/nemotron-finetune-4bit", tokenizer, save_method = "merged_4bit", token = HF_KEY, private=True)

# Just LoRA adapters
if False:
    model.save_pretrained("nemotron_lora")
    tokenizer.save_pretrained("nemotron_lora")
if False: # Pushing to HF Hub
    model.push_to_hub(f"{HF_USERNAME}/nemotron-lora", token = HF_KEY, private=True)
    tokenizer.push_to_hub(f"{HF_USERNAME}/nemotron-lora", token = HF_KEY, private=True)

# %% [markdown]
# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
#
# Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
#
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# %% [markdown]
# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# %%
# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf("nemotron_finetune", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf(f"{HF_USERNAME}/nemotron-finetune", tokenizer, token = HF_KEY, private=True)

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("nemotron_finetune", tokenizer, quantization_method = "f16")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf(f"{HF_USERNAME}/nemotron-finetune", tokenizer, quantization_method = "f16", token = HF_KEY, private=True)

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("nemotron_finetune", tokenizer, quantization_method = "q4_k_m")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf(f"{HF_USERNAME}/nemotron-finetune", tokenizer, quantization_method = "q4_k_m", token = HF_KEY, private=True)

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        f"{HF_USERNAME}/nemotron-finetune", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = HF_KEY, # Get a token at https://huggingface.co/settings/tokens
        private=True,
    )

# %% [markdown]
# Now, use the `nemotron_finetune.Q8_0.gguf` file or `nemotron_finetune.Q4_K_M.gguf` file in llama.cpp.
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
#
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# </div>
