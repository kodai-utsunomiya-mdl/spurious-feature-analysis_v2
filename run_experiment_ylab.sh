#!/bin/bash

# --- YBATCH / Slurm Header Options ---
#YBATCH -r dgx-b200_1

#SBATCH -N 1
#SBATCH -J sp_v2_exp
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err
#SBATCH --time=24:00:00

# --- Environment Setup ---
# 1. Load modules
. /etc/profile.d/modules.sh
module load cuda

# 2. Proxy Settings
export http_proxy=http://10.1.10.1:8080
export https_proxy=http://10.1.10.1:8080
export ftp_proxy=http://10.1.10.1:8080

# 3. Setup uv PATH
# uvがインストールされた場所をPATHに追加
export PATH="$HOME/.local/bin:$PATH"

# 4. Setup Python Environment
echo "Setting up venv..."
# uvコマンドが見つかるか確認
if ! command -v uv &> /dev/null; then
    echo "Error: uv command not found. Please run: source \$HOME/.local/bin/env"
    exit 1
fi

if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate

# 5. Install Dependencies
echo "Installing dependencies..."
uv pip install "numba>=0.59.0" torch torchvision numpy pandas matplotlib pyyaml wandb scikit-learn umap-learn cuml-cu12 --extra-index-url https://pypi.nvidia.com

# --- Main Execution ---
echo "Starting python main.py..."
python main.py

echo "Job finished."
