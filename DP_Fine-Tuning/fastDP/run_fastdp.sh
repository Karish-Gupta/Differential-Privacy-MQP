#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=32g                     # allocate 32 GB of memory
#SBATCH -J "run_fastDP"              # name of the job
#SBATCH -o fastdp_run_%j.out         # name of the output file
#SBATCH -e fastdp_run_%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 02:00:00                   # time limit of 2 hours
#SBATCH --gres=gpu:H200:1             # request 1 H200 GPU

module load python/3.10.2/mqmlxcf
module load cuda/12.4.0/3mdaov5

source env/bin/activate

pip install -U "huggingface_hub[cli]"
pip install --upgrade pip
pip install git+https://github.com/awslabs/fast-differential-privacy.git
pip install numpy
pip install torch
pip install transformers
pip install datasets
pip install tqdm
pip install scikit-learn
pip install accelerate
pip install peft

python fastdp.py
