#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=32g                     # allocate 32 GB of memory
#SBATCH -J "run_fastDP"              # name of the job
#SBATCH -o fastdp_run_%j.out         # name of the output file
#SBATCH -e fastdp_run_%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 12:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H100:2             # request 1 H200 GPU

cd $SLURM_SUBMIT_DIR/..

module load python/3.10.2/mqmlxcf
module load cuda/12.4.0/3mdaov5

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install git+https://github.com/awslabs/fast-differential-privacy.git
pip install -U "huggingface_hub[cli]"
pip install numpy
pip install torch
pip install transformers
pip install datasets
pip install tqdm
pip install scikit-learn
pip install accelerate
pip install peft

accelerate launch \
  --multi_gpu \
  --mixed_precision=fp16 \
  --num_processes 2 \
  --num_machines 1 \
  -m fast_dp.fastdp