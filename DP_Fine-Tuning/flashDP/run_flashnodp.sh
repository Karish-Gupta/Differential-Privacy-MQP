#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=32g                     # allocate 32 GB of memory
#SBATCH -J "run_noDP"              # name of the job
#SBATCH -o flashnodp_run_%j.out         # name of the output file
#SBATCH -e flashnodp_run_%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 01:00:00                   # time limit of 1 hour
#SBATCH --gres=gpu:H200:2             # request 2 H200 GPU

module load python/3.11.10
module load cuda/12.4.0/3mdaov5

python3 -m venv pytorch_example_env
source pytorch_example_env/bin/activate

pip3 install --upgrade pip
pip3 install numpy
pip3 install torch
pip3 install transformers
pip3 install datasets
pip3 install tqdm
pip3 install scikit-learn
pip3 install sentencepiece
pip3 install accelerate

python3 flash_nodp.py
