#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=32g                     # allocate 32 GB of memory
#SBATCH -J "tune_fastDP"              # name of the job
#SBATCH -o fastdp_tune_%j.out         # name of the output file
#SBATCH -e fastdp_tune_%j.err         # name of the error file
#SBATCH -p long                      # partition to submit to
#SBATCH -t 48:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H200:1             # request 1 H200 GPU

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

python -m fast_dp.fastdp

