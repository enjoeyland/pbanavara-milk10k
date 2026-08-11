#!/bin/bash
#SBATCH --job-name=milk10k-cbm
#SBATCH --partition=suma_a6000,gigabyte_a6000
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=cs-gpu-01,node24
#SBATCH --output=/home/khmin1104/workspace/Medical-CausalInference/baselines/pbanavara-milk10k/logs/slurm-%j.out
#SBATCH --error=/home/khmin1104/workspace/Medical-CausalInference/baselines/pbanavara-milk10k/logs/slurm-%j.err

set -euo pipefail

REPO="/home/khmin1104/workspace/Medical-CausalInference/baselines/pbanavara-milk10k"
VENV="/scratch2/khmin1104/venvs/pbanavara-milk10k"

# Redirect torch.hub cache (DINOv2 backbone download, ~1.2GB) off $HOME onto scratch
export TORCH_HOME="/scratch2/khmin1104/cache"
mkdir -p "${TORCH_HOME}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO}"

echo "host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)"

python -m src.train --config configs/default.yaml

echo "done"
