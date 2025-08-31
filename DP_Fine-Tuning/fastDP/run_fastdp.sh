#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00

module load cuda/12.1
module load anaconda
conda activate fastdp-env

torchrun --nproc_per_node=4 train.py
