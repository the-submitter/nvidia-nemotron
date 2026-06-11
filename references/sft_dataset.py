# %%
from __future__ import annotations
import os
from pathlib import Path

# User secrets
try:
    from kaggle_secrets import UserSecretsClient    # type: ignore

    user_secrets = UserSecretsClient()
    GITPAT = user_secrets.get_secret("GITPAT")
    KAGGLE_KEY = user_secrets.get_secret("KAGGLE_KEY")
    KAGGLE_USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
    HF_KEY = user_secrets.get_secret("HF_KEY")

    if KAGGLE_KEY:
        os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    if KAGGLE_USERNAME:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    if HF_KEY:
        os.environ["HF_TOKEN"] = HF_KEY

except:
    GITPAT = os.environ.get("GITPAT")
    KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
    KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
    HF_KEY = os.environ.get("HF_KEY")

DOMAIN = os.environ.get("DOMAIN", "code")
DATA_CFG_PATH = Path(os.environ.get("DATA_CFG_PATH", f"train/domains/{DOMAIN}/sft_dataset.json"))
HF_UPLOAD_USERNAME = os.environ.get("HF_UPLOAD_USERNAME", "the-submitter")

HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/tmp/hf_cache")
KAGGLE_CACHE_DIR = os.environ.get("KAGGLE_CACHE_DIR", "/tmp/kagglehub")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", os.cpu_count()))
DATASET_TRAIN_SIZE = int(os.environ.get("DATASET_TRAIN_SIZE", 50_000))
TEST_VALIDATION_SPLIT = float(os.environ.get("TEST_VALIDATION_SPLIT", 0.1))

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

# %%
# %%capture
# !uv pip install -qqq datasets
# !uv pip install -qqq -r {DATA_CFG_PATH.with_name("requirements.txt")} --torch-backend auto

# %%
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict, Value, load_from_disk
from dataclasses import dataclass
from typing import Optional, Literal, Callable
import os

DATASET_SIZE = round(DATASET_TRAIN_SIZE / (1 - TEST_VALIDATION_SPLIT * 2))
KEEP_IN_MEMORY: Optional[bool] = True

if KEEP_IN_MEMORY:
    # Redirect downloads, models, and lock files directly into RAM
    os.environ["HF_HOME"] = "/dev/shm/huggingface"
    os.environ["HF_DATASETS_CACHE"] = "/dev/shm/huggingface/datasets"

    # Also configure Hugging Face to keep processed tables in memory
    import datasets
    datasets.config.IN_MEMORY_MAX_SIZE = 300 * 1024 * 1024 * 1024  # 300 GB    

@dataclass
class DatasetConfig:
    name: str
    subset: Optional[str] = None
    source: Literal["huggingface", "kaggle", "local"] = "huggingface"
    split: Optional[str] = "train"
    size: Optional[int | float] = None
    streaming: bool = False
    shuffle_seed: int = 42
    shuffle_buffer: int = 10_000
    filter_key: Optional[str] = "response"
    filter_by_length: Optional[Literal["max", "min"]] = None
    filter_map: Optional[dict[str, set]] = None
    filter_map_include: bool = True
    column_map: Optional[dict[str, str]] = None
    column_type_map: Optional[dict[str, Value]] = None
    remove_extra_columns: bool = True
    processor: Optional[Callable[[Dataset, Optional[int], Optional[bool]], Dataset]] = None

def filter_dataset_by_length(dataset: Dataset, data_cfg: DatasetConfig) -> Dataset:
    if data_cfg.filter_key is None or data_cfg.filter_by_length is None:
        return dataset

    if data_cfg.filter_key not in dataset.column_names:
        return dataset

    dataset = dataset.filter(
        lambda example: example.get(data_cfg.filter_key) is not None, 
        num_proc=MAX_WORKERS,
        desc=f"Filter `{data_cfg.filter_key}`: non-None",
        keep_in_memory=KEEP_IN_MEMORY,
    )

    dataset = dataset.map(
        lambda example: {"__filter_length": len(example[data_cfg.filter_key])},
        remove_columns=[],
        num_proc=MAX_WORKERS,
        desc=f"Add temporary length column",
        keep_in_memory=KEEP_IN_MEMORY,
    )

    dataset = dataset.sort(
        "__filter_length", 
        reverse=(data_cfg.filter_by_length == "max"),
        keep_in_memory=KEEP_IN_MEMORY,
    )
    dataset = dataset.remove_columns(["__filter_length"])

    return dataset

def filter_dataset_by_map(dataset: Dataset, data_cfg: DatasetConfig) -> Dataset:
    if data_cfg.filter_map is None:
        return dataset
    
    def _filter(example):
        for filter_key, filter_values in data_cfg.filter_map.items():
            if filter_key not in dataset.column_names:
                continue
            if (not data_cfg.filter_map_include) ^ (example[filter_key] in filter_values):
                return True
        return False
    
    return dataset.filter(
        _filter, 
        num_proc=MAX_WORKERS, 
        desc="Filter by `filter_map`",
        keep_in_memory=KEEP_IN_MEMORY,
    )

def apply_column_map(dataset: Dataset, data_cfg: DatasetConfig) -> Dataset:
    if data_cfg.column_map:
        rename_map, create_map = {}, {}
        for old, new in data_cfg.column_map.items():
            if old in dataset.column_names:
                rename_map[old] = new
            else:
                create_map[old] = new
        if not rename_map and not create_map:
            return dataset

        if data_cfg.remove_extra_columns:
            remove_cols = [col for col in dataset.column_names if col not in rename_map]
            if remove_cols:
                dataset = dataset.remove_columns(remove_cols)

        if rename_map:
            # rename existing columns
            dataset = dataset.rename_columns(rename_map)

        if create_map:
            # add new columns
            def add_columns_(batch):
                batch_size = len(batch[next(iter(batch))])

                for column in create_map.values():
                    batch[column] = [None] * batch_size

                return batch
            
            dataset = dataset.map(
                add_columns_, 
                batched=True, 
                num_proc=MAX_WORKERS,
                desc="Add new columns",
                keep_in_memory=KEEP_IN_MEMORY,
            )
        
    if data_cfg.column_type_map:
        new_features = dataset.features.copy()
        new_features.update(data_cfg.column_type_map)
        dataset = dataset.cast(
            new_features, 
            num_proc=MAX_WORKERS,
            keep_in_memory=KEEP_IN_MEMORY,
        )

    return dataset

def get_dataset(data_cfg: DatasetConfig) -> Dataset:
    if data_cfg.size is None:
        data_cfg.streaming = False
    elif isinstance(data_cfg.size, float) and 0.0 < data_cfg.size < 1.0:
        data_cfg.size *= DATASET_SIZE

    match(data_cfg.source):
        case "huggingface":
            dataset = load_dataset(
                data_cfg.name,
                data_cfg.subset,
                split=data_cfg.split,
                streaming=data_cfg.streaming,
                token=HF_KEY,
                num_proc=MAX_WORKERS,
                keep_in_memory=KEEP_IN_MEMORY,
            )

            if data_cfg.streaming:
                dataset = Dataset.from_generator(
                    lambda: dataset.shuffle(seed=data_cfg.shuffle_seed, buffer_size=data_cfg.shuffle_buffer).take(data_cfg.size),
                    keep_in_memory=KEEP_IN_MEMORY,
                    cache_dir=HF_CACHE_DIR,
                    num_proc=MAX_WORKERS,
                )
            else:
                dataset = dataset.shuffle(seed=data_cfg.shuffle_seed, keep_in_memory=KEEP_IN_MEMORY)
                if data_cfg.size is None:
                    data_cfg.size = len(dataset)
                else:
                    dataset = dataset.take(data_cfg.size)

        case "kaggle":
            import kagglehub

            data_cfg.streaming = False
            dataset_path = f"{KAGGLE_CACHE_DIR}/{data_cfg.name.split('/')[-1]}"

            actual_dataset_path = kagglehub.dataset_download(data_cfg.name, output_dir=dataset_path)
            if actual_dataset_path.startswith("/kaggle/input/") and not os.path.exists(dataset_path):
                print(f"Copying data from {actual_dataset_path!r} to {dataset_path!r} ...")
                import shutil
                shutil.copytree(actual_dataset_path, dataset_path, dirs_exist_ok=True)
            dataset = load_from_disk(dataset_path, keep_in_memory=KEEP_IN_MEMORY)

            if data_cfg.split is not None:
                dataset = dataset[data_cfg.split]

            dataset = dataset.shuffle(seed=data_cfg.shuffle_seed, keep_in_memory=KEEP_IN_MEMORY)

            if data_cfg.size is None:
                data_cfg.size = len(dataset)
            else:
                dataset = dataset.take(data_cfg.size)

        case "local":
            dataset = load_from_disk(data_cfg.name, keep_in_memory=KEEP_IN_MEMORY)

            if data_cfg.split is not None:
                dataset = dataset[data_cfg.split]

            dataset = dataset.shuffle(seed=data_cfg.shuffle_seed, keep_in_memory=KEEP_IN_MEMORY)

            if data_cfg.size is None:
                data_cfg.size = len(dataset)
            else:
                dataset = dataset.take(data_cfg.size)

        case _: raise ValueError(f"DatasetConfig.source = {data_cfg.source} not supported")

    dataset = apply_column_map(dataset, data_cfg)

    return dataset

def prepare_dataset(data_cfg: DatasetConfig) -> Dataset:
    dataset = get_dataset(data_cfg)
    dataset = filter_dataset_by_length(dataset, data_cfg)
    dataset = filter_dataset_by_map(dataset, data_cfg)
    return data_cfg.processor(dataset, MAX_WORKERS, KEEP_IN_MEMORY) if data_cfg.processor is not None else dataset

def build_final_dataset(
        datasets_list: list[Dataset], 
        seed: int = 42, 
        processor: Optional[Callable[[Dataset, Optional[int], Optional[bool]], Dataset]] = None,
) -> Dataset | DatasetDict:
    total_records = sum(len(ds) for ds in datasets_list)
    datasets_list = [
        ds.take(min(len(ds), round((len(ds) / total_records) * DATASET_SIZE))) 
        for ds in datasets_list
    ]
    non_empty = [dataset for dataset in datasets_list if len(dataset) > 0]
    if not non_empty:
        raise ValueError("No datasets available to build the final dataset.")

    final_dataset = concatenate_datasets(non_empty) if len(non_empty) > 1 else non_empty[0]

    if TEST_VALIDATION_SPLIT:
        train_split = final_dataset.train_test_split(
            test_size=TEST_VALIDATION_SPLIT * 2, 
            seed=seed,
            keep_in_memory=KEEP_IN_MEMORY,
        )
        temp_split = train_split["test"].train_test_split(
            test_size=0.5, 
            seed=seed,
            keep_in_memory=KEEP_IN_MEMORY,
        )

        final_dataset = DatasetDict({
            "train": train_split["train"],
            "validation": temp_split["train"],
            "test": temp_split["test"]
        })
    else:
        final_dataset = final_dataset.shuffle(seed=seed, keep_in_memory=KEEP_IN_MEMORY)
    
    return processor(final_dataset, MAX_WORKERS, KEEP_IN_MEMORY) if processor is not None else final_dataset

# %%
import json
import sys
import importlib
if str(repo_dir) not in sys.path:
    sys.path.append(str(repo_dir))
utils = importlib.import_module(str(DATA_CFG_PATH.with_name("utils")).replace("/", "."))

with open(DATA_CFG_PATH, "r") as f:
    data_cfgs_ = json.load(f)

data_cfgs = []
for data_cfg in data_cfgs_:
    column_type_map = data_cfg.pop("column_type_map", None)
    if column_type_map is not None:
        for k in column_type_map:
            column_type_map[k] = Value(column_type_map[k])

    processor = data_cfg.pop("processor", None)
    if processor:
        processor = getattr(utils, processor, None)

    if "filter_map" in data_cfg and data_cfg["filter_map"]:
        for fk in data_cfg["filter_map"]:
            data_cfg["filter_map"][fk] = set(data_cfg["filter_map"][fk])

    data_cfgs.append(
        DatasetConfig(
            **data_cfg, 
            column_type_map=column_type_map, 
            processor=processor,
        )
    )

print(data_cfgs)

# %%
datasets = [prepare_dataset(data_cfg) for data_cfg in data_cfgs]

print(datasets)

# %%
utils = importlib.import_module(f"train.domains.{DOMAIN}.utils")
process_dataset = getattr(utils, "process_dataset", None)

final_dataset = build_final_dataset(
    datasets, 
    seed=data_cfgs[0].shuffle_seed, 
    processor=process_dataset,
)

print(final_dataset)

# %%
dataset_tag = f"{DOMAIN}-{int(DATASET_SIZE // 1000)}k".replace("_", "-")

# Save to huggingface
final_dataset.save_to_disk(f"{HF_CACHE_DIR}/{dataset_tag}", num_proc=MAX_WORKERS)
final_dataset.push_to_hub(f"{HF_UPLOAD_USERNAME}/{dataset_tag}", private=True, token=HF_KEY)

# Save to kaggle
import os
import kagglehub

# 1. Setup kaggle parameters and save locally
kaggle_folder = f"{KAGGLE_CACHE_DIR}/{dataset_tag}"

# Save your dataset splits as parquet files
os.makedirs(kaggle_folder, exist_ok=True)
for split_name, dataset in final_dataset.items():
    dataset.to_parquet(os.path.join(kaggle_folder, f"{split_name}.parquet"))

# 2. Upload using kagglehub
# Format: 'username/dataset-slug'
handle = f"{KAGGLE_USERNAME}/{dataset_tag}"

kagglehub.dataset_upload(
    handle=handle,
    local_dataset_dir=kaggle_folder,
)
