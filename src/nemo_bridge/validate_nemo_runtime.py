"""Validate the offline Python runtime used by the NeMo-RL subprocess."""

from __future__ import annotations

import sys
from importlib.metadata import version as installed_version

import ray  # noqa: F401
import torch
import transformers
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def validate_nemo_runtime() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Expected Python 3.12 for the Kaggle wheel set, got {sys.version}"
        )
    if Version(transformers.__version__) not in SpecifierSet(">=5.5.0,<5.6.0"):
        raise RuntimeError(
            "This NeMo-RL/AutoModel checkout requires Transformers >=5.5,<5.6, "
            f"got {transformers.__version__}"
        )
    antlr_version = Version(installed_version("antlr4-python3-runtime"))
    if antlr_version != Version("4.9.3"):
        raise RuntimeError(
            "OmegaConf 2.3.0 requires antlr4-python3-runtime==4.9.3; "
            f"got {antlr_version}. Rerun the Kaggle Dependencies cell to repair "
            "the offline virtual environment."
        )
    try:
        import vllm
    except Exception as exc:
        raise RuntimeError(
            "The bundled vLLM wheel could not load with Kaggle's Torch "
            "2.10/CUDA 12.8 runtime. Upstream vLLM 0.20 release wheels default "
            "to CUDA 13, so the offline wheelhouse must include the compatible "
            "wheel/runtime set."
        ) from exc

    import nemo_automodel  # noqa: F401
    import nemo_rl  # noqa: F401
    import nemotron_nemo_bridge  # noqa: F401

    print(
        "Validated NeMo runtime:",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
        f"transformers={transformers.__version__}",
        f"vllm={vllm.__version__}",
    )
    if torch.__version__.split("+", 1)[0] != "2.10.0":
        raise RuntimeError(
            "Expected torch 2.10.0 for the Kaggle wheel set, "
            f"got {torch.__version__}"
        )
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected CUDA 12.8 PyTorch, got {torch.version.cuda}")
    if Version(vllm.__version__).release[:2] != (0, 20):
        raise RuntimeError(
            "This NeMo-RL checkout requires the bundled vLLM 0.20.x API, "
            f"got {vllm.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the NeMo-RL subprocess")
    torch.empty(1, device="cuda")
    print("CUDA device:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    validate_nemo_runtime()
