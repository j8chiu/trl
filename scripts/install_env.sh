#!/usr/bin/env bash
set -euo pipefail

# Run inside your conda/venv environment.
# For CUDA 12.x + RTX 3090, adjust the torch wheel index if your driver requires another CUDA minor version.

pip install --upgrade pip
pip install --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade "accelerate>=0.34" "datasets" "transformers" "peft" "trackio" "rich" "tqdm" "math-verify"

# Recommended: install TRL from source because the distillation trainer is experimental and changes quickly.
# If you already cloned TRL, run this from the TRL repo root instead:
#   pip install -e ".[peft]"
pip install --upgrade "trl[peft]"

accelerate config default
