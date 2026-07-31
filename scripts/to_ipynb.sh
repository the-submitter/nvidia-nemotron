#!/bin/bash

./scripts/jupytext_convert.sh src --to-ipynb --destination notebooks/ --exclude src/nemo_bridge --exclude __init__.py --maintain-structure
