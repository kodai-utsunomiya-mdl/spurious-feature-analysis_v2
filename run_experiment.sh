#!/bin/bash

# --- SGEスケジューラへの指示 (qrshの引数をここで指定) ---
#$ -S /bin/bash
#$ -jc gpu-container_g1_dev
#$ -ac d=nvcr-pytorch-2503

# --- ジョブの管理 ---
#$ -N sp_v2_batch_job             # ジョブ名を指定
#$ -cwd                           # このスクリプトがあるディレクトリ(~/sp_v2)で実行
#$ -o logs/job.o$JOB_ID           # 標準出力(print文など)の保存先
#$ -e logs/job.e$JOB_ID           # 標準エラーの保存先

# --- 実行するコマンド (ここからがジョブ本体) ---

# 0. 環境設定の読み込み (uvコマンドなどを見つけるため)
source ~/.bashrc

# 1. プロキシ設定
export http_proxy=http://10.1.10.1:8080
export https_proxy=http://10.1.10.1:8080
export ftp_proxy=http://10.1.10.1:8080

# 2. 仮想環境の準備
# (毎回--clearすると時間がかかるため，初回以降はコメントアウト推奨)
echo "Setting up venv..."
uv venv --clear
source .venv/bin/activate

# 3. 依存関係のインストール
# (同上，初回以降はコメントアウト推奨)
echo "Installing dependencies..."
uv pip install torch torchvision numpy pandas matplotlib pyyaml wandb

# 4. メインスクリプトの実行
echo "Starting python main.py..."
python main.py

echo "Job finished."
