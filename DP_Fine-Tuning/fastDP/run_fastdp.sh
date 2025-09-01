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

module load python/3.10.0
module load cuda/12.4.0/3mdaov5

python3 -m venv env
source env/bin/activate

pip install git+https://github.com/awslabs/fast-differential-privacy.git
pip3 install --upgrade pip
pip3 install numpy
pip3 install torch
pip3 install transformers
pip3 install datasets
pip3 install tqdm
pip3 install scikit-learn
pip3 install accelerate

python3 fastdp.py
