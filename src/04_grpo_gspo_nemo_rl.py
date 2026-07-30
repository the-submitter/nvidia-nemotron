# %% [markdown]
# ## NeMo-RL GSPO/GRPO Overview
# - Runs GSPO/GRPO-style training through NeMo-RL with JSONL response data, a custom
#   reward environment, and colocated vLLM generation.
# - Builds reward-ready prompts with source, HQ, DPO-aware, and remaining-source ordering
#   controls.
# - Combines exact answer verification, fuzzy matching, completion similarity, boxed
#   formatting, and think-tag rewards.
# - Keeps the existing dataset preparation and reward logic while avoiding Unsloth/vLLM
#   synchronization issues on the Nemotron hybrid model.

# %% [markdown]
# ## Imports
# - Load dependencies for this notebook script.
# - Set early runtime flags before heavier stage-specific imports run.

# %%
from __future__ import annotations

import json
import os
import shutil
import site
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")



# %% [markdown]
# ## Credentials
# - Read Kaggle, Hugging Face, and W&B credentials when available.
# - Keep committed defaults safe for public or local dry runs.

# %%
# User secrets
# try:
#     from kaggle_secrets import UserSecretsClient  # type: ignore
#
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

HF_KEY = WANDB_KEY = KAGGLE_KEY = KAGGLE_USERNAME = None



# %% [markdown]
# ## Kaggle Dependencies
# - Discover the `unsloth-vllm-wheels-temp` Kaggle input without relying on one
#   mount spelling.
# - Build an isolated Python 3.12 environment that reuses Kaggle's CUDA 12.8
#   PyTorch and installs every compatible cached artifact with network access disabled.
#   Kaggle's Ubuntu Python omits `ensurepip`, so the environment is created with
#   `--without-pip` and reuses its system `pip` via `--system-site-packages`.
# - Use the bundled NeMo-RL source directly. `uv run` is intentionally not used:
#   NeMo-RL's worker-specific `uv` environments can otherwise try to resolve or
#   build packages that are unavailable on an internet-disabled Kaggle session.
# - The bundled RDMA `.deb` files are only a fallback when `libibverbs` is absent.
#   Only runtime packages are installed; the development package is unnecessary
#   because cached wheels and source archives are installed locally without a
#   dependency resolver reaching the network.
#   cuDNN archives are not required by this DTensor + vLLM recipe (they are needed
#   by the Megatron backend), so they are deliberately left untouched.

# %%
def _is_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}


def find_kaggle_bundle() -> Path | None:
    configured = os.environ.get("KAGGLE_BUNDLE_DIR")
    candidates = [
        Path(configured) if configured else None,
        Path("/kaggle/input/unsloth-vllm-wheels-temp"),
        Path(
            "/kaggle/input/datasets/rohitraje0493/"
            "unsloth-vllm-wheels-temp"
        ),
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob("*/"))
        candidates.extend(kaggle_input.glob("datasets/*/*/"))

    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if (
            (candidate / "packages").is_dir()
            and (candidate / "nemo-rl" / "examples" / "run_grpo.py").is_file()
            and (candidate / "nvidia-nemotron" / "src").is_dir()
        ):
            return candidate
    return None


def find_code_repo(bundle_dir: Path | None) -> Path:
    configured = os.environ.get("NEMOTRON_CODE_DIR")
    if configured:
        return Path(configured).resolve()
    if bundle_dir is not None:
        return (bundle_dir / "nvidia-nemotron").resolve()
    file_value = globals().get("__file__")
    if file_value:
        file_path = Path(str(file_value)).resolve()
        if file_path.is_file() and file_path.parent.name == "src":
            return file_path.parent.parent
    if (Path.cwd() / "src" / "nemo_bridge").is_dir():
        return Path.cwd().resolve()
    raise FileNotFoundError(
        "Could not find the nvidia-nemotron source tree. Set NEMOTRON_CODE_DIR."
    )


KAGGLE_RUNTIME = (
    "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").is_dir()
)
KAGGLE_BUNDLE_DIR = find_kaggle_bundle()
if KAGGLE_RUNTIME and KAGGLE_BUNDLE_DIR is None:
    raise FileNotFoundError(
        "The Kaggle dependency input was not found. Attach "
        "`unsloth-vllm-wheels-temp` or set KAGGLE_BUNDLE_DIR."
    )

CODE_REPO_DIR = find_code_repo(KAGGLE_BUNDLE_DIR)
CODE_SRC_DIR = CODE_REPO_DIR / "src"
if str(CODE_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_SRC_DIR))

DEFAULT_NEMO_RL_DIR = (
    KAGGLE_BUNDLE_DIR / "nemo-rl"
    if KAGGLE_BUNDLE_DIR is not None
    else Path(
        os.environ.get(
            "LOCAL_NEMO_RL_DIR",
            "/media/rohit-raje/Elements/Rohit/rohit_lenovo/explore/nemo-rl",
        )
    )
)
KAGGLE_NEMO_VENV = Path(
    os.environ.get("KAGGLE_NEMO_VENV", "/kaggle/working/.venv")
)
NEMO_VENV_SITE_PACKAGES = (
    KAGGLE_NEMO_VENV
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
KAGGLE_SYSTEM_RUNTIME_DISTRIBUTIONS = {
    "torch",
    "torchaudio",
    "torchvision",
    "triton",
}
KAGGLE_SYSTEM_CUDA_DISTRIBUTION_PREFIXES = (
    "nvidia-cublas-",
    "nvidia-cuda-",
    "nvidia-cudnn-",
    "nvidia-cufft-",
    "nvidia-cufile-",
    "nvidia-curand-",
    "nvidia-cusolver-",
    "nvidia-cusparse-",
    "nvidia-cusparselt-",
    "nvidia-nccl-",
    "nvidia-nvjitlink-",
    "nvidia-nvshmem-",
)
KAGGLE_WHEEL_VERSION_CONSTRAINTS = {
    # OmegaConf 2.3.0 ships parsers generated by ANTLR 4.9.  Newer ANTLR
    # runtimes cannot deserialize their serialized ATN representation.
    "antlr4-python3-runtime": "==4.9.3",
    "flashinfer-cubin": "==0.6.8.post1",
    "flashinfer-jit-cache": "==0.6.8.post1",
    "flashinfer-python": "==0.6.8.post1",
    "omegaconf": "==2.3.0",
    "transformers": ">=5.5.0,<5.6.0",
    "vllm": ">=0.20.0,<0.21.0",
}
KAGGLE_REQUIRED_RUNTIME_DISTRIBUTIONS = (
    "antlr4-python3-runtime",
    "omegaconf",
)
KAGGLE_WHEEL_MANIFEST_VERSION = 9


def _has_package_build_metadata(package_dir: Path) -> bool:
    return any(
        (package_dir / filename).is_file()
        for filename in ("pyproject.toml", "setup.cfg", "setup.py", "PKG-INFO")
    ) or any(package_dir.glob("*.egg-info/PKG-INFO"))


def _package_install_root(package_dir: Path) -> Path:
    """Handle Kaggle's extra directory around extracted source-package roots."""
    from packaging.utils import canonicalize_name, parse_sdist_filename

    nested_dirs = sorted(
        path
        for path in package_dir.iterdir()
        if path.is_dir() and _has_package_build_metadata(path)
    )
    if not nested_dirs:
        return package_dir

    expected_names = {
        canonicalize_name(package_dir.name),
        canonicalize_name(package_dir.name.split("-", maxsplit=1)[0]),
    }
    try:
        distribution_name, _version = parse_sdist_filename(
            f"{package_dir.name}.tar.gz"
        )
        expected_names.add(canonicalize_name(distribution_name))
    except Exception:
        pass
    for nested_dir in nested_dirs:
        if canonicalize_name(nested_dir.name) in expected_names:
            return nested_dir
    if len(nested_dirs) == 1:
        return nested_dirs[0]
    return package_dir


def _package_directory_metadata(package_dir: Path) -> tuple[str, Any] | None:
    """Read a source tree's distribution metadata without running setup code."""
    import configparser
    import tomllib
    from email.parser import Parser
    from packaging.version import Version

    pyproject_path = package_dir / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            project = tomllib.loads(pyproject_path.read_text()).get("project", {})
            name = project.get("name")
            version = project.get("version")
            if isinstance(name, str) and isinstance(version, str):
                return name, Version(version)
        except (OSError, ValueError, tomllib.TOMLDecodeError):
            pass

    metadata_paths = [package_dir / "PKG-INFO"]
    metadata_paths.extend(package_dir.glob("*.egg-info/PKG-INFO"))
    for metadata_path in metadata_paths:
        try:
            metadata = Parser().parsestr(metadata_path.read_text())
            name = metadata.get("Name")
            version = metadata.get("Version")
            if name and version:
                return name, Version(version)
        except (OSError, ValueError):
            continue

    setup_cfg_path = package_dir / "setup.cfg"
    if setup_cfg_path.is_file():
        try:
            setup_cfg = configparser.ConfigParser()
            setup_cfg.read(setup_cfg_path)
            name = setup_cfg.get("metadata", "name")
            version = setup_cfg.get("metadata", "version")
            return name, Version(version)
        except (configparser.Error, ValueError):
            pass
    return None


def _package_artifact_metadata(
    package_path: Path,
    supported_tags: dict[Any, int] | None = None,
) -> tuple[str, Any, int, int] | None:
    """Return distribution metadata and rank for a wheel, archive, or source tree."""
    from email.parser import Parser
    from packaging.utils import parse_sdist_filename, parse_wheel_filename

    try:
        if package_path.is_file() and package_path.suffix == ".whl":
            name, version, _build, wheel_tags = parse_wheel_filename(
                package_path.name
            )
            if supported_tags is None:
                return name, version, 3, 0
            tag_ranks = [
                supported_tags[tag] for tag in wheel_tags if tag in supported_tags
            ]
            if not tag_ranks:
                return None
            return name, version, 3, -min(tag_ranks)
        if package_path.is_file() and package_path.suffix == ".bin":
            # Kaggle automatically extracts .zip/.whl uploads.  A wheel renamed
            # to .bin remains a zip file, so preserve its original wheel name
            # semantics while selecting it.
            try:
                name, version, _build, wheel_tags = parse_wheel_filename(
                    package_path.with_suffix(".whl").name
                )
                if supported_tags is None:
                    return name, version, 3, 0
                tag_ranks = [
                    supported_tags[tag]
                    for tag in wheel_tags
                    if tag in supported_tags
                ]
                if tag_ranks:
                    return name, version, 3, -min(tag_ranks)
                # It is definitely a renamed wheel, but not one this Python
                # can install. Do not accidentally accept its METADATA below.
                return None
            except Exception:
                pass

            # A renamed source zip may not retain a parseable wheel filename.
            # Read standard package metadata from the zip without extracting it
            # into Kaggle's read-only input mount.
            with zipfile.ZipFile(package_path) as package_zip:
                metadata_names = sorted(
                    name
                    for name in package_zip.namelist()
                    if name.endswith("/PKG-INFO")
                    or name.endswith(".dist-info/METADATA")
                )
                for metadata_name in metadata_names:
                    metadata = Parser().parsestr(
                        package_zip.read(metadata_name).decode(
                            "utf-8", errors="replace"
                        )
                    )
                    name, version = metadata.get("Name"), metadata.get("Version")
                    if name and version:
                        from packaging.version import Version

                        return name, Version(version), 2, 0
            # Last chance for a source archive whose filename was merely given
            # a .bin extension.
            name, version = parse_sdist_filename(
                f"{package_path.stem}.tar.gz"
            )
            return name, version, 2, 0
        if package_path.is_file():
            name, version = parse_sdist_filename(package_path.name)
            return name, version, 2, 0
        if package_path.is_dir():
            metadata = _package_directory_metadata(package_path)
            if metadata is not None:
                name, version = metadata
                return name, version, 1, 0
            name, version = parse_sdist_filename(f"{package_path.name}.tar.gz")
            return name, version, 1, 0
    except Exception:
        return None
    return None


def _bundle_source_artifacts(bundle_dir: Path) -> list[Path]:
    """Find source-package directories placed beside the `packages` folder."""
    excluded_names = {"packages", "nemo-rl", "nvidia-nemotron"}
    artifacts = []
    for path in bundle_dir.iterdir():
        if not path.is_dir() or path.name in excluded_names:
            continue
        if _has_package_build_metadata(path) or any(
            child.is_dir() and _has_package_build_metadata(child)
            for child in path.iterdir()
        ):
            artifacts.append(path)
    return artifacts


def _compatible_wheels(
    wheel_dir: Path,
    extra_artifacts: tuple[Path, ...] = (),
) -> list[Path]:
    """Select compatible artifacts, preferring wheels, archives, then folders."""
    from packaging.specifiers import SpecifierSet
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name

    supported_tags = {tag: index for index, tag in enumerate(sys_tags())}
    selected: dict[str, tuple[int, Any, int, Path]] = {}
    artifact_paths = [*wheel_dir.iterdir(), *extra_artifacts]
    for package_path in sorted(artifact_paths):
        install_path = (
            _package_install_root(package_path)
            if package_path.is_dir()
            else package_path
        )
        metadata = _package_artifact_metadata(install_path, supported_tags)
        if metadata is None:
            continue
        name, version, artifact_kind_rank, tag_rank = metadata
        key = canonicalize_name(name)
        version_constraint = KAGGLE_WHEEL_VERSION_CONSTRAINTS.get(key)
        if (
            version_constraint is not None
            and version not in SpecifierSet(version_constraint)
        ):
            continue
        candidate = (artifact_kind_rank, version, tag_rank, install_path)
        if key not in selected or candidate[:3] > selected[key][:3]:
            selected[key] = candidate
    missing_required = [
        name for name in KAGGLE_REQUIRED_RUNTIME_DISTRIBUTIONS if name not in selected
    ]
    if missing_required:
        requirements = ", ".join(
            f"{name}{KAGGLE_WHEEL_VERSION_CONSTRAINTS[name]}"
            for name in missing_required
        )
        raise FileNotFoundError(
            "The offline package cache is missing required NeMo-RL runtime artifacts: "
            f"{requirements}. Rebuild `unsloth-vllm-wheels-temp/packages` with "
            "the lock-compatible package artifacts."
        )
    return [selected[name][3] for name in sorted(selected)]


def _matching_system_distribution(wheel_path: Path) -> bool:
    """Reuse exact matches and preserve Kaggle's CUDA 12.8 Torch runtime."""
    from importlib.metadata import PackageNotFoundError, version as installed_version
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    metadata = _package_artifact_metadata(wheel_path)
    if metadata is None:
        return False
    name, candidate_version, _artifact_kind_rank, _tag_rank = metadata
    try:
        system_version = Version(installed_version(name))
    except (PackageNotFoundError, ValueError):
        return False
    normalized_name = canonicalize_name(name)
    if (
        normalized_name in KAGGLE_SYSTEM_RUNTIME_DISTRIBUTIONS
        or normalized_name.startswith(KAGGLE_SYSTEM_CUDA_DISTRIBUTION_PREFIXES)
    ):
        return True
    return system_version == candidate_version


def ensure_rdma_runtime(bundle_dir: Path) -> None:
    import ctypes.util

    if ctypes.util.find_library("ibverbs"):
        print("RDMA runtime already provides libibverbs; local .deb files skipped")
        return
    deb_names = [
        "rdma-core_39.0-1_amd64.deb",
        "libibverbs1_39.0-1_amd64.deb",
        "ibverbs-providers_39.0-1_amd64.deb",
    ]
    deb_paths = [bundle_dir / name for name in deb_names]
    missing = [str(path) for path in deb_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing offline RDMA packages: {missing}")
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        raise PermissionError(
            "libibverbs is missing and installing the bundled .deb files "
            "requires a root Kaggle session"
        )
    subprocess.run(["dpkg", "-i", *map(str, deb_paths)], check=True)


def nemo_bootstrap_env() -> dict[str, str]:
    """Keep NeMo's `sitecustomize` out of venv and pip bootstrap processes.

    `src/nemo_bridge` deliberately contains a sitecustomize module for the
    eventual NeMo/Ray subprocesses. If it leaks in through a notebook's
    PYTHONPATH while the venv is being created, it imports packages that have
    not been installed yet and can break Python before `venv` starts.
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def ensure_nemo_venv_pip(python_path: Path, venv_dir: Path) -> None:
    """Require the system pip exposed through the no-ensurepip virtualenv."""
    pip_check = subprocess.run(
        [str(python_path), "-m", "pip", "--version"],
        env=nemo_bootstrap_env(),
        text=True,
        capture_output=True,
    )
    if pip_check.returncode != 0:
        details = (pip_check.stderr or pip_check.stdout).strip()
        raise RuntimeError(
            "The Kaggle Python image must provide pip for the offline NeMo "
            "environment. Its Python venv module has no ensurepip, so the "
            "bootstrap intentionally reuses system pip through "
            "--system-site-packages. "
            f"Could not run pip in {venv_dir}: {details}"
        )


def install_offline_artifacts(
    python_path: Path,
    venv_dir: Path,
    artifacts: list[Path],
) -> None:
    """Install read-only Kaggle artifacts without letting pip mutate source trees."""
    with tempfile.TemporaryDirectory(
        prefix=".nemo-offline-build-",
        dir=venv_dir,
    ) as staging_dir:
        staging_path = Path(staging_dir)
        install_paths: list[Path] = []
        for index, artifact_path in enumerate(artifacts):
            if artifact_path.is_file() and artifact_path.suffix == ".bin":
                extracted_path = staging_path / f"{index}-{artifact_path.stem}"
                try:
                    with zipfile.ZipFile(artifact_path) as package_zip:
                        package_zip.extractall(extracted_path)
                except zipfile.BadZipFile as exc:
                    raise RuntimeError(
                        f"Offline package {artifact_path} has a .bin suffix but "
                        "is not a zip file. Rename or replace it with a valid "
                        "wheel/source archive."
                    ) from exc

                # A .bin file is normally a wheel whose extension was changed
                # to prevent Kaggle auto-extraction. Pip needs the .whl suffix
                # to install that layout, even though we also unpack it above.
                try:
                    from packaging.utils import parse_wheel_filename

                    parse_wheel_filename(artifact_path.with_suffix(".whl").name)
                except Exception:
                    source_root = _package_install_root(extracted_path)
                    if not _has_package_build_metadata(source_root):
                        nested_wheels = sorted(extracted_path.rglob("*.whl"))
                        if len(nested_wheels) == 1:
                            install_paths.append(nested_wheels[0])
                            continue
                        raise RuntimeError(
                            f"Extracted {artifact_path} into {extracted_path}, "
                            "but could not find a pip-installable source root or "
                            "wheel."
                        )
                    install_paths.append(source_root)
                else:
                    staged_wheel_path = staging_path / (
                        f"{index}-{artifact_path.stem}.whl"
                    )
                    shutil.copy2(artifact_path, staged_wheel_path)
                    install_paths.append(staged_wheel_path)
                continue

            if not artifact_path.is_dir():
                install_paths.append(artifact_path)
                continue
            staged_source_path = staging_path / f"{index}-{artifact_path.name}"
            shutil.copytree(artifact_path, staged_source_path)
            install_paths.append(staged_source_path)

        command = [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--disable-pip-version-check",
            *map(str, install_paths),
        ]
        install_env = nemo_bootstrap_env()
        install_env["PIP_NO_CACHE_DIR"] = "1"
        subprocess.run(command, env=install_env, check=True)


def create_offline_nemo_environment(
    bundle_dir: Path,
    venv_dir: Path,
) -> Path:
    wheel_dir = bundle_dir / "packages"
    root_source_artifacts = _bundle_source_artifacts(bundle_dir)
    if (
        not wheel_dir.is_dir()
        or (not any(wheel_dir.iterdir()) and not root_source_artifacts)
    ):
        raise FileNotFoundError(
            f"No dependency artifacts found under {wheel_dir} or {bundle_dir}"
        )

    python_path = venv_dir / "bin" / "python"
    venv_config_path = venv_dir / "pyvenv.cfg"
    if not python_path.is_file() or not venv_config_path.is_file():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "venv",
                "--without-pip",
                "--system-site-packages",
                str(venv_dir),
            ],
            env=nemo_bootstrap_env(),
            check=True,
        )
    ensure_nemo_venv_pip(python_path, venv_dir)

    compatible_wheels = _compatible_wheels(
        wheel_dir,
        tuple(root_source_artifacts),
    )
    manifest_path = venv_dir / ".nemotron-wheel-manifest.json"
    manifest = {
        "format": KAGGLE_WHEEL_MANIFEST_VERSION,
        "artifacts": [str(path.relative_to(bundle_dir)) for path in compatible_wheels],
    }
    if manifest_path.is_file():
        try:
            if json.loads(manifest_path.read_text()) == manifest:
                print(f"Reusing populated offline wheel environment at {venv_dir}")
                return python_path
        except (OSError, json.JSONDecodeError):
            pass

    install_artifacts = [
        path
        for path in compatible_wheels
        if not _matching_system_distribution(path)
    ]
    if install_artifacts:
        print(
            "Installing "
            f"{len(install_artifacts)} compatible offline artifacts into {venv_dir}"
        )
        install_offline_artifacts(python_path, venv_dir, install_artifacts)
    else:
        print(f"Offline wheel environment is already populated at {venv_dir}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return python_path


NEMO_PYTHON = Path(sys.executable)
if KAGGLE_RUNTIME and _is_truthy(
    os.environ.get("KAGGLE_BOOTSTRAP_NEMO_DEPS"),
    default=True,
):
    assert KAGGLE_BUNDLE_DIR is not None
    ensure_rdma_runtime(KAGGLE_BUNDLE_DIR)
    NEMO_PYTHON = create_offline_nemo_environment(
        KAGGLE_BUNDLE_DIR,
        KAGGLE_NEMO_VENV,
    )
    if NEMO_VENV_SITE_PACKAGES.is_dir():
        site.addsitedir(str(NEMO_VENV_SITE_PACKAGES))
        if str(NEMO_VENV_SITE_PACKAGES) in sys.path:
            sys.path.remove(str(NEMO_VENV_SITE_PACKAGES))
        sys.path.insert(0, str(NEMO_VENV_SITE_PACKAGES))

print(f"NeMo-RL source: {DEFAULT_NEMO_RL_DIR}")
print(f"NeMo-RL Python: {NEMO_PYTHON}")



# %% [markdown]
# ## Runtime Configuration
# - Define paths, split names, source controls, hyperparameters, and upload destinations.
# - Read values from environment variables so Kaggle and local runs can override defaults.

# %%
WORKING_DIR = Path(os.environ.get("WORKING_DIR", "/kaggle/working"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    # "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/default/1",
)
ADAPTER_INPUT_PATH = os.environ.get(
    "ADAPTER_INPUT_PATH",
    None,
)
BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/kaggle/input/datasets/rohitraje0493/nemotron-reasoning",
)
DATASET_REVISION = os.environ.get("DATASET_REVISION")
HF_CACHE_DIR = Path(os.environ.get("HF_CACHE_DIR", "/tmp/hf_cache"))
TRAIN_SPLIT = os.environ.get("TRAIN_SPLIT", "train")
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", None)

def optional_nonnegative_int(
    name: str,
    default: Optional[int] = None,
) -> Optional[int]:
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

def bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}

TRAIN_MIN_IDX = optional_nonnegative_int("TRAIN_MIN_IDX")
TRAIN_MAX_IDX = optional_nonnegative_int("TRAIN_MAX_IDX", 9500)
EVAL_MIN_IDX = optional_nonnegative_int("EVAL_MIN_IDX")
EVAL_MAX_IDX = optional_nonnegative_int("EVAL_MAX_IDX")
SOURCE_OPTIONS = {
    TRAIN_SPLIT: {
        "include": optional_string_list("TRAIN_INCLUDE_SOURCES", '["nvidia-nemotron-model-reasoning-challenge", "dgxchen/nemotron-cot-tong"]'),
        "order": optional_string_list("TRAIN_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("TRAIN_EXCLUDE_SOURCES"),
        "order_remaining": os.environ.get(
            "TRAIN_ORDER_REMAINING",
            "0",
        ).lower() not in {"0", "false", "no"},
        "dpo_sort_remaining": os.environ.get(
            "TRAIN_DPO_SORT_REMAINING",
            "1",
        ).lower() not in {"0", "false", "no"},
    },
    EVAL_SPLIT: {
        "include": optional_string_list("EVAL_INCLUDE_SOURCES"),
        "order": optional_string_list("EVAL_ORDER_BY_SOURCES"),
        "exclude": optional_string_list("EVAL_EXCLUDE_SOURCES"),
        "order_remaining": os.environ.get(
            "EVAL_ORDER_REMAINING",
            "0",
        ).lower() not in {"0", "false", "no"},
        "dpo_sort_remaining": os.environ.get(
            "EVAL_DPO_SORT_REMAINING",
            "1",
        ).lower() not in {"0", "false", "no"},
    },
}
SHUFFLE_BY_SPLIT = {
    TRAIN_SPLIT: os.environ.get("TRAIN_SHUFFLE", "0").lower()
        not in {"0", "false", "no"},
    EVAL_SPLIT: os.environ.get("EVAL_SHUFFLE", "0").lower()
        not in {"0", "false", "no"},
}
FILTER_HQ_BY_SPLIT = {
    TRAIN_SPLIT: os.environ.get("TRAIN_FILTER_HQ", "1").lower()
        not in {"0", "false", "no"},
    EVAL_SPLIT: os.environ.get("EVAL_FILTER_HQ", "1").lower()
        not in {"0", "false", "no"},
}
DPO_AWARE_BY_SPLIT = {
    TRAIN_SPLIT: os.environ.get("TRAIN_DPO_AWARE", "1").lower()
        not in {"0", "false", "no"},
    EVAL_SPLIT: os.environ.get("EVAL_DPO_AWARE", "1").lower()
        not in {"0", "false", "no"},
}

TRAIN_STAGE = os.environ.get("TRAIN_STAGE", "gspo")
TRAIN_VERSION = os.environ.get("TRAIN_VERSION", "v1")
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"nemotron-{TRAIN_STAGE}-{TRAIN_VERSION}",
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKING_DIR / RUN_NAME)))
ADAPTER_OUTPUT_DIR = Path(
    os.environ.get(
        "ADAPTER_OUTPUT_DIR",
        str(
            WORKING_DIR
            / f"nemotron-lora-{TRAIN_STAGE}-{TRAIN_VERSION}"
        ),
    )
)
HF_USERNAME = os.environ.get("HF_USERNAME", "the-submitter")
HF_ADAPTER_REPO = os.environ.get(
    "HF_ADAPTER_REPO",
    f"{HF_USERNAME}/nemotron-lora-{TRAIN_STAGE}-{TRAIN_VERSION}",
)
KAGGLE_ADAPTER_REPO = os.environ.get(
    "KAGGLE_ADAPTER_REPO",
    f"{KAGGLE_USERNAME}/nemotron-3-nano/transformers/lora-{TRAIN_STAGE}",
)
KAGGLE_DATASET_REPO = os.environ.get(
    "KAGGLE_DATASET_REPO",
    f"{KAGGLE_USERNAME}/nemotron-{TRAIN_STAGE}",
)

DATASET_WORKERS = max(1, int(os.environ.get("DATASET_NUM_PROC", "8")))
DATASET_NUM_PROC = DATASET_WORKERS if DATASET_WORKERS > 1 else None
SEED = int(os.environ.get("SEED", "3407"))

MATH_VERIFY_TIMEOUT_SECONDS = int(
    os.environ.get("MATH_VERIFY_TIMEOUT_SECONDS", "5")
)
if MATH_VERIFY_TIMEOUT_SECONDS <= 0:
    raise ValueError("MATH_VERIFY_TIMEOUT_SECONDS must be positive")

EXACT_MATCH_WEIGHT = float(os.environ.get("EXACT_MATCH_WEIGHT", "5.0"))
ANSWER_FUZZY_WEIGHT = float(os.environ.get("ANSWER_FUZZY_WEIGHT", "3.0"))
COMPLETION_FUZZY_WEIGHT = float(
    os.environ.get("COMPLETION_FUZZY_WEIGHT", "0.15")
)
BOXED_WEIGHT = float(os.environ.get("BOXED_WEIGHT", "1.0"))
THINK_WEIGHT = float(os.environ.get("THINK_WEIGHT", "0.25"))

REPORT_TO = os.environ.get("REPORT_TO", "wandb")
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

if REPORT_TO == "wandb":
    os.environ.setdefault("WANDB_MODE", "offline")
    wandb_dir = Path(os.environ.get("WANDB_DIR", str(WORKING_DIR / "wandb_logs")))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ.setdefault("WANDB_SILENT", "true")




# %% [markdown]
# ## Helper Functions
# - Define reusable helpers including `clean_text`, `is_high_quality_example`,
#   `is_reward_example`, `build_user_content`, `build_grpo_example`, `dpo_priority`.
# - Support parsing, filtering, formatting, verification, loading, or upload behavior used
#   later.

# %%
from nemo_bridge.nemotron_reward_utils import (  # noqa: E402
    clean_text,
    combine_reasoning_response,
    unified_reward,
)

HQ_SOURCES = {
    "nvidia-nemotron-model-reasoning-challenge",
    "dgxchen/nemotron-cot-tong",
}

HQ_ANSWER_TYPES = {"integer", "float", "fraction"}

def is_high_quality_example(example: dict[str, Any]) -> bool:
    if example.get("response") and not example.get("reasoning"):
        return False
    if example.get("source") in HQ_SOURCES:
        return True
    answer_type = clean_text(example.get("answer_type"))
    if answer_type is not None and answer_type.lower() in HQ_ANSWER_TYPES:
        return True
    return False
    # final_answer = clean_text(example.get("final_answer"))
    # return final_answer is not None and final_answer.isalnum()


def is_reward_example(example: dict[str, Any]) -> bool:
    return (
        clean_text(example.get("prompt")) is not None
        and clean_text(example.get("final_answer")) is not None
    )


def build_user_content(prompt: Any) -> str:
    normalized_prompt = clean_text(prompt)
    if normalized_prompt is None:
        raise ValueError("Cannot format GRPO data without a prompt")
    return normalized_prompt + BOXED_ANSWER_INSTRUCTION


def build_grpo_example(example: dict[str, Any]) -> dict[str, Any]:
    final_answer = clean_text(example.get("final_answer"))
    if final_answer is None:
        raise ValueError("Dataset must be filtered before GRPO formatting")
    return {
        "prompt": [
            {
                "role": "user",
                "content": build_user_content(example.get("prompt")),
            }
        ],
        "response": clean_text(example.get("response")),
        "reasoning": clean_text(example.get("reasoning")),
        "final_answer": final_answer,
    }


def dpo_priority(example: dict[str, Any]) -> int:
    chosen = clean_text(example.get("chosen"))
    rejected = clean_text(example.get("rejected"))
    selected = bool(example.get("dpo_selected"))
    if rejected is not None and chosen is None:
        return 0
    if selected and chosen is not None and rejected is not None:
        return 1
    if selected:
        return 2
    return 3




# %% [markdown]
# ## Dataset Loading
# - Load datasets from local disk, Kaggle-mounted paths, Hugging Face repos, parquet files,
#   or saved DatasetDicts.
# - Normalize split handling and cache behavior for downstream processing.

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


def apply_source_options(dataset, split_name: str):
    options = SOURCE_OPTIONS.get(
        split_name,
        {"include": [], "order": [], "exclude": []},
    )
    include_sources = options["include"]
    order_sources = options["order"]
    exclude_sources = set(options["exclude"])
    order_remaining = options.get("order_remaining", False)
    dpo_sort_remaining = options.get("dpo_sort_remaining", True)
    dpo_aware = DPO_AWARE_BY_SPLIT.get(split_name)
    if (
        not include_sources
        and not order_sources
        and not exclude_sources
        and not dpo_aware
    ):
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
        before_remaining_sources = (
            include_sources
            + [source for source in explicit_order if source in before_wildcard]
        )
        after_remaining_sources = [
            source for source in explicit_order if source not in before_wildcard
        ]
    else:
        before_remaining_sources = include_sources + explicit_order
        after_remaining_sources = []
    ordered_sources = (
        before_remaining_sources
        + (remaining_sources if order_remaining else ["*"])
        + after_remaining_sources
    )

    indices_by_source: dict[Any, list[int]] = {}
    for index, source in enumerate(dataset["source"]):
        indices_by_source.setdefault(source, []).append(index)

    def ordered_source_indices(source: Any) -> list[int]:
        source_indices = indices_by_source.get(source, [])
        if dpo_aware:
            source_indices = sorted(
                source_indices,
                key=lambda index: dpo_priority(dataset[index]),
            )
        return source_indices

    before_remaining_indices = [
        index
        for source in before_remaining_sources
        for index in ordered_source_indices(source)
    ]
    if order_remaining:
        remaining_indices = [
            index
            for source in remaining_sources
            for index in ordered_source_indices(source)
        ]
    else:
        remaining_source_set = set(remaining_sources)
        if dpo_aware and dpo_sort_remaining:
            remaining_indices_by_source = {
                source: iter(ordered_source_indices(source))
                for source in remaining_sources
            }
            remaining_indices = [
                next(remaining_indices_by_source[source])
                for source in dataset["source"]
                if source in remaining_source_set
            ]
        else:
            remaining_indices = [
                index
                for index, source in enumerate(dataset["source"])
                if source in remaining_source_set
            ]
    after_remaining_indices = [
        index
        for source in after_remaining_sources
        for index in ordered_source_indices(source)
    ]
    selected_indices = (
        before_remaining_indices
        + remaining_indices
        + after_remaining_indices
    )

    ordered = dataset.select(selected_indices, keep_in_memory=KEEP_IN_MEMORY)
    print(
        f"{split_name}: source order={ordered_sources}; "
        f"order_remaining={order_remaining}; dpo_aware={dpo_aware}; "
        f"dpo_sort_remaining={dpo_sort_remaining}; "
        f"retained={len(ordered):,}"
    )
    return ordered


def prepare_split(
    dataset,
    split_name: str,
    allow_empty: bool = False,
):
    from datasets import Features, List, Value

    original_size = len(dataset)
    dataset = dataset.filter(
        is_reward_example,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: keep reward-ready examples",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    if FILTER_HQ_BY_SPLIT.get(split_name, False):
        before_hq = len(dataset)
        dataset = dataset.filter(
            is_high_quality_example,
            num_proc=DATASET_NUM_PROC,
            desc=f"{split_name}: keep high-quality examples",
            keep_in_memory=KEEP_IN_MEMORY,
        )
        print(
            f"{split_name}: HQ filter retained "
            f"{len(dataset):,}/{before_hq:,} examples"
        )
    if not len(dataset):
        if allow_empty:
            print(f"{split_name}: no reward-ready examples after filtering")
            return None
        raise ValueError(f"{split_name} has no reward-ready examples")

    grpo_features = Features(
        {
            "prompt": List(
                {
                    "role": Value("string"),
                    "content": Value("string"),
                }
            ),
            "response": Value("string"),
            "reasoning": Value("string"),
            "final_answer": Value("string"),
        }
    )
    dataset = dataset.map(
        build_grpo_example,
        remove_columns=dataset.column_names,
        features=grpo_features,
        num_proc=DATASET_NUM_PROC,
        desc=f"{split_name}: build GRPO conversations",
        keep_in_memory=KEEP_IN_MEMORY,
    )
    print(f"{split_name}: retained {len(dataset):,}/{original_size:,} examples")
    return dataset


def select_index_range(
    dataset,
    min_idx: Optional[int],
    max_idx: Optional[int],
    split_name: str,
    shuffle: bool,
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
        else dataset.select(
            range(start, stop),
            keep_in_memory=KEEP_IN_MEMORY,
        )
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
        selected = selected.shuffle(
            seed=SEED,
            keep_in_memory=KEEP_IN_MEMORY,
        )
        print(f"{split_name}: shuffled {len(selected):,} examples with seed {SEED}")
    return selected


def prepare_datasets():
    dataset_dict = load_reasoning_dataset()
    train_source_dataset = apply_source_options(
        dataset_dict[TRAIN_SPLIT],
        TRAIN_SPLIT,
    )
    train_dataset = prepare_split(
        train_source_dataset,
        TRAIN_SPLIT,
    )
    train_dataset = select_index_range(
        train_dataset,
        TRAIN_MIN_IDX,
        TRAIN_MAX_IDX,
        TRAIN_SPLIT,
        SHUFFLE_BY_SPLIT[TRAIN_SPLIT],
    )

    eval_dataset = None
    if (
        EVAL_SPLIT
        and EVAL_SPLIT in dataset_dict
        and len(dataset_dict[EVAL_SPLIT])
    ):
        eval_source_dataset = apply_source_options(
            dataset_dict[EVAL_SPLIT],
            EVAL_SPLIT,
        )
        eval_dataset = prepare_split(
            eval_source_dataset,
            EVAL_SPLIT,
            allow_empty=True,
        )
        if eval_dataset is not None:
            eval_dataset = select_index_range(
                eval_dataset,
                EVAL_MIN_IDX,
                EVAL_MAX_IDX,
                EVAL_SPLIT,
                SHUFFLE_BY_SPLIT[EVAL_SPLIT],
            )
    return train_dataset, eval_dataset




# %% [markdown]
# ## Adapter Input Preparation
# - NeMo-RL's DTensor GRPO path creates and checkpoints its own LoRA modules.
# - It does not accept an Unsloth/Hugging Face PEFT adapter directory as
#   `policy.model_name`. Start from `MODEL_PATH`, or point `ADAPTER_INPUT_PATH`
#   at a previously merged, complete Hugging Face checkpoint.

# %%
def prepare_adapter_input_path() -> Optional[str]:
    if ADAPTER_INPUT_PATH is None:
        return None

    source_path = Path(ADAPTER_INPUT_PATH)
    if not source_path.is_dir():
        raise FileNotFoundError(
            f"ADAPTER_INPUT_PATH does not exist: {source_path}"
        )
    full_weight_files = (
        list(source_path.glob("model*.safetensors"))
        + list(source_path.glob("pytorch_model*.bin"))
    )
    if not (source_path / "config.json").is_file() or not full_weight_files:
        raise RuntimeError(
            "ADAPTER_INPUT_PATH is an adapter-only checkpoint. NeMo-RL "
            "DTensor GRPO cannot continue directly from an Unsloth/PEFT "
            "adapter. Merge it with the base model in a separate internet-"
            "enabled preparation job, attach that full checkpoint to Kaggle, "
            "or unset ADAPTER_INPUT_PATH to train a fresh NeMo-RL LoRA."
        )
    return str(source_path.resolve())




# %% [markdown]
# ## NeMo-RL JSONL Settings
# - Configure response-dataset JSONL locations consumed by NeMo-RL.
# - Keep backend-specific knobs isolated from the common dataset and reward code.

# %%
NEMO_DATA_DIR = Path(os.environ.get("NEMO_DATA_DIR", WORKING_DIR / "nemo_rl_data"))
NEMO_TRAIN_JSONL = NEMO_DATA_DIR / "train.jsonl"
NEMO_EVAL_JSONL = NEMO_DATA_DIR / "validation.jsonl"
NEMO_RL_DIR = Path(
    os.environ.get("NEMO_RL_DIR", str(DEFAULT_NEMO_RL_DIR))
).resolve()
NEMO_CONFIG_PATH = Path(
    os.environ.get(
        "NEMO_CONFIG_PATH",
        str(CODE_REPO_DIR / "configs" / "grpo_gspo_nemotron.yaml"),
    )
)
NEMO_RUNTIME_CONFIG_PATH = Path(
    os.environ.get(
        "NEMO_RUNTIME_CONFIG_PATH",
        str(WORKING_DIR / "grpo_gspo_nemotron.runtime.yaml"),
    )
)
NEMO_BRIDGE_DIR = (CODE_SRC_DIR / "nemo_bridge").resolve()
NEMO_RUNTIME_VALIDATOR = NEMO_BRIDGE_DIR / "validate_nemo_runtime.py"
NEMO_RUN_TRAIN = bool_env("NEMO_RUN_TRAIN", True)


def nemo_source_workspaces() -> list[Path]:
    """Return editable NeMo workspace dependencies normally installed by uv."""
    candidates = [
        NEMO_RL_DIR / "3rdparty" / "Automodel-workspace" / "Automodel",
        NEMO_RL_DIR / "3rdparty" / "Gym-workspace" / "Gym",
    ]
    return [path.resolve() for path in candidates if path.is_dir()]




# %% [markdown]
# ## NeMo-RL JSONL Export
# - Convert prepared Hugging Face rows into NeMo-RL `ResponseDataset` JSONL records.
# - Preserve prompt, reference answer, reasoning, and response metadata for the reward environment.

# %%
def nemo_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        user_contents = [
            str(message.get("content") or "")
            for message in prompt
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if user_contents:
            return user_contents[-1]
    raise TypeError(f"Unsupported prompt value for NeMo-RL export: {type(prompt)}")


def nemo_jsonl_record(example: dict[str, Any]) -> dict[str, Any]:
    prompt = nemo_prompt_text(example.get("prompt"))
    reference_completion = combine_reasoning_response(
        example.get("reasoning"),
        example.get("response"),
    )
    return {
        "input": prompt,
        "output": reference_completion,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reference_completion},
        ],
        "final_answer": example.get("final_answer"),
        "reasoning": example.get("reasoning"),
        "response": example.get("response"),
        "source": example.get("source"),
        "answer_type": example.get("answer_type"),
        "dpo_selected": example.get("dpo_selected"),
    }


def write_jsonl_dataset(dataset, output_path: Path, split_name: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in dataset:
            handle.write(json.dumps(nemo_jsonl_record(example), ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset):,} {split_name} examples to {output_path}")
    return output_path


def prepare_nemo_jsonl_datasets(train_dataset, eval_dataset):
    train_jsonl = write_jsonl_dataset(train_dataset, NEMO_TRAIN_JSONL, TRAIN_SPLIT)
    eval_jsonl = None
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_jsonl = write_jsonl_dataset(eval_dataset, NEMO_EVAL_JSONL, EVAL_SPLIT)
    return train_jsonl, eval_jsonl




# %% [markdown]
# ## NeMo-RL Config and Bridge
# - Materialize a writable runtime YAML from the checked-in recipe.
# - Inject Kaggle-local model, data, output, and logging paths so the launch never
#   depends on stale hard-coded paths.
# - Enforce the mutually exclusive GSPO loss switches.

# %%
def resolve_nemo_config_path() -> Path:
    if NEMO_CONFIG_PATH.exists():
        return NEMO_CONFIG_PATH.resolve()
    candidate = (Path.cwd() / NEMO_CONFIG_PATH).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"NeMo config not found: {NEMO_CONFIG_PATH}")


def materialize_nemo_runtime_config(
    template_path: Path,
    train_jsonl: Path,
    eval_jsonl: Path | None,
) -> Path:
    from omegaconf import OmegaConf

    config = OmegaConf.load(template_path)
    adapter_or_model_path = prepare_adapter_input_path()
    model_path = adapter_or_model_path or MODEL_PATH
    model_dir = Path(model_path)
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            "MODEL_PATH must be a complete, locally mounted Hugging Face "
            f"checkpoint on an offline Kaggle run: {model_dir}"
        )

    config.policy.model_name = str(model_dir.resolve())
    config.policy.tokenizer.name = str(model_dir.resolve())
    config.checkpointing.checkpoint_dir = str(ADAPTER_OUTPUT_DIR.resolve())
    config.checkpointing.metric_name = None
    config.logger.log_dir = str((OUTPUT_DIR / "nemo_logs").resolve())
    config.logger.wandb_enabled = bool_env("NEMO_WANDB_ENABLED", False)
    config.logger.tensorboard_enabled = bool_env(
        "NEMO_TENSORBOARD_ENABLED",
        False,
    )
    config.grpo.seed = SEED
    config.grpo.max_num_steps = int(
        os.environ.get("NEMO_MAX_STEPS", config.grpo.max_num_steps)
    )
    config.grpo.num_prompts_per_step = int(
        os.environ.get(
            "NEMO_NUM_PROMPTS_PER_STEP",
            config.grpo.num_prompts_per_step,
        )
    )
    config.grpo.num_generations_per_prompt = int(
        os.environ.get(
            "NEMO_NUM_GENERATIONS_PER_PROMPT",
            config.grpo.num_generations_per_prompt,
        )
    )
    config.checkpointing.save_period = int(
        os.environ.get(
            "NEMO_SAVE_PERIOD",
            config.checkpointing.save_period,
        )
    )
    config.policy.max_total_sequence_length = int(
        os.environ.get(
            "NEMO_MAX_TOTAL_SEQUENCE_LENGTH",
            config.policy.max_total_sequence_length,
        )
    )
    config.data.max_input_seq_length = int(
        os.environ.get(
            "NEMO_MAX_INPUT_SEQUENCE_LENGTH",
            config.data.max_input_seq_length,
        )
    )
    config.policy.generation.max_new_tokens = int(
        os.environ.get(
            "NEMO_MAX_NEW_TOKENS",
            config.policy.generation.max_new_tokens,
        )
    )
    config.policy.generation.vllm_cfg.gpu_memory_utilization = float(
        os.environ.get(
            "NEMO_VLLM_GPU_MEMORY_UTILIZATION",
            config.policy.generation.vllm_cfg.gpu_memory_utilization,
        )
    )
    config.policy.dtensor_cfg.lora_cfg.dim = int(
        os.environ.get(
            "NEMO_LORA_DIM",
            config.policy.dtensor_cfg.lora_cfg.dim,
        )
    )
    config.policy.dtensor_cfg.lora_cfg.alpha = int(
        os.environ.get(
            "NEMO_LORA_ALPHA",
            config.policy.dtensor_cfg.lora_cfg.alpha,
        )
    )

    config.data.train.data_path = str(train_jsonl.resolve())
    if eval_jsonl is None:
        config.data.validation = None
        config.grpo.val_at_start = False
        config.grpo.val_at_end = False
    else:
        config.data.validation = {
            "data_path": str(eval_jsonl.resolve()),
            "dataset_name": "ResponseDataset",
            "input_key": "input",
            "output_key": "output",
            "processor": "nemotron_grpo_data_processor",
            "env_name": "nemotron_unified_reward",
        }

    use_gspo = TRAIN_STAGE.strip().lower() == "gspo"
    config.loss_fn.sequence_level_importance_ratios = bool_env(
        "NEMO_SEQUENCE_LEVEL_IMPORTANCE_RATIOS",
        use_gspo,
    )
    config.loss_fn.token_level_loss = bool_env(
        "NEMO_TOKEN_LEVEL_LOSS",
        not config.loss_fn.sequence_level_importance_ratios,
    )
    if (
        config.loss_fn.sequence_level_importance_ratios
        and config.loss_fn.token_level_loss
    ):
        raise ValueError(
            "GSPO sequence-level importance ratios are mutually exclusive "
            "with token-level loss. Set NEMO_TOKEN_LEVEL_LOSS=0."
        )

    expected_global_batch = (
        int(config.grpo.num_prompts_per_step)
        * int(config.grpo.num_generations_per_prompt)
    )
    config.policy.train_global_batch_size = int(
        os.environ.get(
            "NEMO_TRAIN_GLOBAL_BATCH_SIZE",
            expected_global_batch,
        )
    )
    if int(config.policy.train_global_batch_size) != expected_global_batch:
        raise ValueError(
            "policy.train_global_batch_size must equal "
            "grpo.num_prompts_per_step * grpo.num_generations_per_prompt "
            f"for this on-policy recipe ({expected_global_batch})"
        )
    # if (
    #     int(config.data.max_input_seq_length)
    #     + int(config.policy.generation.max_new_tokens)
    #     > int(config.policy.max_total_sequence_length)
    # ):
    #     raise ValueError(
    #         "NEMO_MAX_INPUT_SEQUENCE_LENGTH + NEMO_MAX_NEW_TOKENS "
    #         "must not exceed NEMO_MAX_TOTAL_SEQUENCE_LENGTH"
    #     )
    if not (
        0.0
        < float(config.policy.generation.vllm_cfg.gpu_memory_utilization)
        < 1.0
    ):
        raise ValueError(
            "NEMO_VLLM_GPU_MEMORY_UTILIZATION must be between 0 and 1"
        )

    NEMO_RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, NEMO_RUNTIME_CONFIG_PATH)
    print(f"Materialized NeMo-RL runtime config at {NEMO_RUNTIME_CONFIG_PATH}")
    return NEMO_RUNTIME_CONFIG_PATH.resolve()


def validate_nemo_config_paths(config_path: Path, train_jsonl: Path, eval_jsonl: Path | None) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    configured_train = Path(str(config.data.train.data_path)).resolve()
    actual_train = train_jsonl.resolve()
    if configured_train != actual_train:
        raise ValueError(
            "NeMo config train data_path does not match exported JSONL: "
            f"{configured_train} != {actual_train}. Update {config_path}."
        )
    if config.data.validation is None:
        if eval_jsonl is not None:
            print("Validation JSONL was exported, but NeMo config validation is null")
        return
    if eval_jsonl is None:
        raise ValueError(
            f"NeMo config expects validation data at {config.data.validation.data_path}, "
            "but no eval dataset was exported. Set validation: null or configure EVAL_SPLIT."
        )
    configured_eval = Path(str(config.data.validation.data_path)).resolve()
    actual_eval = eval_jsonl.resolve()
    if configured_eval != actual_eval:
        raise ValueError(
            "NeMo config validation data_path does not match exported JSONL: "
            f"{configured_eval} != {actual_eval}. Update {config_path}."
        )


def validate_nemo_bridge() -> None:
    bridge_module = NEMO_BRIDGE_DIR / "nemotron_nemo_bridge.py"
    sitecustomize_module = NEMO_BRIDGE_DIR / "sitecustomize.py"
    if not all(
        path.exists()
        for path in (
            bridge_module,
            sitecustomize_module,
            NEMO_RUNTIME_VALIDATOR,
        )
    ):
        raise FileNotFoundError(
            f"Expected NeMo bridge files under {NEMO_BRIDGE_DIR}"
        )




# %% [markdown]
# ## Training Runtime Bootstrap
# - Materialize train and validation splits, then export them to NeMo-RL JSONL files.
# - Check that the static NeMo config and bridge module are ready for subprocess launch.

# %%
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

train_dataset, eval_dataset = prepare_datasets()
train_jsonl, eval_jsonl = prepare_nemo_jsonl_datasets(train_dataset, eval_dataset)
nemo_config_template_path = resolve_nemo_config_path()
nemo_config_path = materialize_nemo_runtime_config(
    nemo_config_template_path,
    train_jsonl,
    eval_jsonl,
)
validate_nemo_config_paths(nemo_config_path, train_jsonl, eval_jsonl)
validate_nemo_bridge()

print("NeMo-RL GRPO/GSPO training sample:")
print(train_dataset[0])
sample_reward = unified_reward(
    prompts=[train_dataset[0]["prompt"]],
    completions=[
        (
            "<think>\nExample reasoning\n</think>\n"
            f"\\boxed{{{train_dataset[0]['final_answer']}}}"
        )
    ],
    response=[train_dataset[0]["response"]],
    reasoning=[train_dataset[0]["reasoning"]],
    final_answer=[train_dataset[0]["final_answer"]],
)
print(f"Reward sanity check: {sample_reward}")




# %% [markdown]
# ## NeMo-RL Subprocess Launch
# - Run the checked-in driver with the isolated offline Python environment.
# - Put `src/nemo_bridge` on `PYTHONPATH` so `sitecustomize` registers the processor and environment.
# - Force all Ray actors to reuse that environment; this avoids NeMo-RL trying
#   to create worker-specific `uv` environments from the network.

# %%
def nemo_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    python_paths = (
        [str(NEMO_VENV_SITE_PACKAGES)]
        if NEMO_VENV_SITE_PACKAGES.is_dir()
        else []
    )
    python_paths.extend(
        [
        str(NEMO_BRIDGE_DIR),
        str(NEMO_RL_DIR),
        *(str(path) for path in nemo_source_workspaces()),
        str(CODE_SRC_DIR),
        ]
    )
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["NEMO_VENV_SITE_PACKAGES"] = str(NEMO_VENV_SITE_PACKAGES)
    env["PYTHONNOUSERSITE"] = "1"
    env["NEMO_RL_PY_EXECUTABLES_SYSTEM"] = "1"
    env.setdefault(
        "NEMO_KAGGLE_BF16_LORA",
        "1" if KAGGLE_RUNTIME else "0",
    )
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["UV_OFFLINE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = str(HF_CACHE_DIR)
    env["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("TRANSFORMERS_NO_FLAX", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    return env


def validate_nemo_python() -> None:
    subprocess.run(
        [str(NEMO_PYTHON), str(NEMO_RUNTIME_VALIDATOR)],
        cwd=str(NEMO_RL_DIR),
        env=nemo_subprocess_env(),
        check=True,
        text=True,
    )


def run_nemo_grpo_subprocess(config_path: Path) -> subprocess.CompletedProcess[str] | None:
    if not NEMO_RUN_TRAIN:
        print("Skipping NeMo-RL training because NEMO_RUN_TRAIN=0")
        return None
    if not NEMO_RL_DIR.exists():
        raise FileNotFoundError(f"NEMO_RL_DIR does not exist: {NEMO_RL_DIR}")
    if not (NEMO_RL_DIR / "examples" / "run_grpo.py").exists():
        raise FileNotFoundError(f"examples/run_grpo.py not found under {NEMO_RL_DIR}")
    # validate_nemo_python()
    command = [
        str(NEMO_PYTHON),
        "examples/run_grpo.py",
        "--config",
        str(config_path),
    ]
    print(f"Running NeMo-RL subprocess in {NEMO_RL_DIR}: {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=str(NEMO_RL_DIR),
        env=nemo_subprocess_env(),
        check=True,
        text=True,
    )


nemo_subprocess_result = run_nemo_grpo_subprocess(nemo_config_path)




# %% [markdown]
# ## Adapter Metadata Normalization
# - Rewrite saved adapter metadata back from Kaggle-local model paths to the original base model id.
# - Keep uploaded adapter configs portable outside the Kaggle runtime.

# %%
if MODEL_PATH and BASE_MODEL_ID:
    for metadata_path in [
        ADAPTER_OUTPUT_DIR / "README.md",
        ADAPTER_OUTPUT_DIR / "adapter_config.json",
    ]:
        if metadata_path.exists():
            metadata_path.write_text(
                metadata_path.read_text().replace(MODEL_PATH, BASE_MODEL_ID)
            )

print(f"NeMo-RL checkpoints/adapters are under {ADAPTER_OUTPUT_DIR}")




# %% [markdown]
# ## Upload Artifacts
# - Publish NeMo-RL checkpoint or adapter folders when upload flags are enabled.
# - Use folder uploads because NeMo-RL owns checkpoint serialization.

# %%
if PUSH_TO_HUB:
    try:
        if not HF_KEY:
            raise RuntimeError("PUSH_TO_HUB=1 but HF_KEY/HF_TOKEN is not configured")
        from huggingface_hub import HfApi, login

        login(token=HF_KEY)
        api = HfApi()
        api.create_repo(
            repo_id=HF_ADAPTER_REPO,
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=ADAPTER_OUTPUT_DIR,
            repo_id=HF_ADAPTER_REPO,
            repo_type="model",
        )
        print("Upload to Hugging Face succeeded")
    except Exception as exc:
        print(f"Upload to Hugging Face failed: {exc}")




# %% [markdown]
# ## Upload Artifacts
# - Publish adapter/checkpoint folders to Kaggle when requested.
# - Remove transient local Kaggle state after successful model upload.

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
            version_notes=f"Nemotron LoRA continued with NeMo-RL {TRAIN_STAGE}",
            license_name="Apache 2.0",
        )
        (WORKING_DIR / "state.db").unlink(missing_ok=True)
        print("Upload to Kaggle succeeded")
    except Exception as exc:
        print(f"Upload to Kaggle failed: {exc}")
