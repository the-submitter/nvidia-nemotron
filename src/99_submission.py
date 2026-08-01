# %%
import subprocess
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")
ADAPTER_OUTPUT_DIR = "/kaggle/input/models/rohitraje0493/nemotron-3-nano/transformers/lora-dpo/9"
SUBMISSION_FILE = WORKING_DIR / "submission.zip"

# !rm -rf {SUBMISSION_FILE}

subprocess.run(f"zip -r {SUBMISSION_FILE} *", cwd=ADAPTER_OUTPUT_DIR, shell=True, check=True)
