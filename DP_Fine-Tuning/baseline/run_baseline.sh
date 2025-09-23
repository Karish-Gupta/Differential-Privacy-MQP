#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 1                          # total number of tasks
#SBATCH --mem=32g                     # allocate 32 GB of memory
#SBATCH -J "run_baseline"              # name of the job
#SBATCH -o baseline_run_%j.out         # name of the output file
#SBATCH -e baseline_run_%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 12:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H100:1             # request 1 H100 GPU

cd $SLURM_SUBMIT_DIR/..

module load python/3.11.10
module load cuda/12.4.0/3mdaov5

python -m venv env
source env/bin/activate

pip install --upgrade pip
pip install numpy
pip install peft
pip install torch
pip install transformers
pip install datasets
pip install tqdm
pip install scikit-learn
pip install sentencepiece
pip install accelerate

python -m baseline.baseline