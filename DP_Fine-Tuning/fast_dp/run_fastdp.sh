#!/bin/bash
#SBATCH -N 1                          # allocate 1 compute node
#SBATCH -n 2                          # total number of tasks (one per GPU)
#SBATCH --mem=141g                    # allocate 141 GB of memory
#SBATCH -J "run_fastDP_MP"           # name of the job
#SBATCH -o fastdp_run_%j.out         # name of the output file
#SBATCH -e fastdp_run_%j.err         # name of the error file
#SBATCH -p short                      # partition to submit to
#SBATCH -t 12:00:00                   # time limit of 12 hours
#SBATCH --gres=gpu:H200:2             # request 2 H200 GPUs
#SBATCH --ntasks-per-node=2           # tasks per node (one per GPU)

cd $SLURM_SUBMIT_DIR/..

module load python/3.10.2/mqmlxcf
module load cuda/12.4.0/3mdaov5

# Create and activate virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    python -m venv env
fi
source env/bin/activate

# Install dependencies
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
pip install deepspeed

# Run with DeepSpeed using model and tensor parallelism
deepspeed --num_gpus 2 DP_Fine-Tuning/fast_dp/fastdp.py \
  --deepspeed \
  --deepspeed_config DP_Fine-Tuning/fast_dp/ds_config.json \
  --master_port $(shuf -i 10000-65535 -n 1)