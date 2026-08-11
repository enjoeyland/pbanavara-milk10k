#!/bin/bash
#SBATCH --job-name=chexpert-pbanavara-cbm
#SBATCH --partition=asus_a5000,gigabyte_a5000,suma_rtx4090,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_a6000,gigabyte_a6000
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --exclude=cs-gpu-01,node24
#SBATCH --output=/home/khmin1104/workspace/Medical-CausalInference/baselines/pbanavara-milk10k/logs/slurm-%j.out
#SBATCH --error=/home/khmin1104/workspace/Medical-CausalInference/baselines/pbanavara-milk10k/logs/slurm-%j.err

set -euo pipefail
REPO="/home/khmin1104/workspace/Medical-CausalInference/baselines/pbanavara-milk10k"
VENV="/scratch2/khmin1104/venvs/pbanavara-milk10k"
export TORCH_HOME="/scratch2/khmin1104/cache"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO}"

echo "host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python train_chexpert.py "$@"
echo "done"
