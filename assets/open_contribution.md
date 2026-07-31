# NVIDIA Nemotron Reasoning Challenge Contribution

Repository: <https://github.com/the-submitter/nvidia-nemotron>

## Dataset

The dataset was built in two phases: 
- first, a consolidated reasoning dataset was created from roughly 15 sources; 
- then it was updated with DPO preference fields (`chosen` and `rejected` completions).

### Phase 1: Create Dataset

Notebook: <https://www.kaggle.com/code/rohitraje0493/nvidia-nemotron-create-dataset>

- Builds the canonical Nemotron reasoning `DatasetDict` used by the SFT,
preference-generation, DPO, and GRPO/GSPO stages. 
- Normalizes Hugging Face, Kaggle, and local sources into a shared schema 
containing prompts, reasoning, responses, final answers, and metadata. 
- Supports streaming selection, length ranking, source quotas, per-dataset 
and global deduplication, high-quality filtering, deterministic split shuffling, 
schema validation, and optional publication to Hugging Face and Kaggle.

Sources include NuminaMath, competition_math, OpenR1-Math, GSM8K, SVAMP, ASDiv,
DROP, ProofWriter, FOLIO, ProntoQA, ZebraLogic, Enigmata, OpenMathReasoning,
Nemotron COT Tong, and the NVIDIA Nemotron Model Reasoning Challenge data.

### Phase 2: Update Dataset

Notebook: <https://www.kaggle.com/code/rohitraje0493/nvidia-nemotron-update-dataset>

- Generates DPO preference fields for prompts that need `chosen` and `rejected`
completions. 
- Selects candidates with source ordering, high-quality filters,
range/take limits, and resumable bookkeeping. 
- Runs vLLM trajectories.
- Verifies final answers. 
- Selects short correct and incorrect completions. 
- Writes incremental snapshots, optional CSV backups, and upload-ready datasets.

## Supervised Fine-Tuning

Notebook: <https://www.kaggle.com/code/rohitraje0493/nvidia-nemotron-sft>

- SFT uses Unsloth over the consolidated dataset.
- Filters response-ready rows, applies source and high-quality controls, 
formats chat examples, creates or resumes a LoRA adapter, and trains with TRL `SFTTrainer`.
- The run saves trainer state, adapter weights, tokenizer files, and optional Hugging Face/Kaggle
artifacts.

## Direct Preference Optimization

Notebook: <https://www.kaggle.com/code/rohitraje0493/nvidia-nemotron-dpo>

- DPO continues the SFT LoRA adapter with the preference-updated dataset.
- It filters rows with `chosen`/`rejected` fields, formats prompt-completion pairs,
uses Unsloth and TRL `DPOTrainer`, and saves the continued adapter with optional
published artifacts.

## Experimental GSPO/GRPO

### Unsloth

Notebook: <https://www.kaggle.com/code/rohitraje0493/nvidia-nemotron-grpo-gspo-unsloth?scriptVersionId=329178155>

- Experiments with GSPO/GRPO using Unsloth, TRL `GRPOTrainer`, and colocated vLLM.
- Combines source/HQ/DPO-aware ordering with answer correctness, completion
similarity, boxed-format, and think-tag rewards.
- Kaggle runs encountered Unsloth/vLLM state-dict synchronization issues
- The Transformers fallback was too slow.

### Native TRL

Notebook: <https://www.kaggle.com/code/rohitraje0493/nvidia-nemotron-grpo-gspo-trl>

- Uses native Transformers, PEFT, TRL, and colocated vLLM with the same dataset
preparation, ordering controls, reward logic, and artifact flow. 
- Recorded Kaggle runs hit CUDA out-of-memory errors.

### NeMo RL

Git Branch: [`nemo-rl`](https://github.com/the-submitter/nvidia-nemotron/tree/nemo-rl)

Notebook: <https://www.kaggle.com/code/rohitraje0493/nvidia-nemotron-grpo-gspo-nemo-rl>

- The NeMo-RL path runs GSPO/GRPO through NeMo-RL's `run_grpo.py` entry point.
- Prepares the same reward-ready prompts and unified answer/format/reasoning
reward, registers a custom NeMo-RL reward environment, and uses a custom GRPO
configuration with colocated vLLM generation.
- It is tailored to an internet-disabled Kaggle RTX GPU server by installing 
cached dependencies into a writable offline virtual environment while reusing 
Kaggle's CUDA-matched PyTorch. 
- This installation/runtime integration remains experimental.
