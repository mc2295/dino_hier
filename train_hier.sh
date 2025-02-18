#!/bin/bash

#SBATCH -o log/out_hier.txt
#SBATCH -e log/err_hier.txt
#SBATCH -J dino_hier
#SBATCH -p gpu_p
#SBATCH --mem=300G
#SBATCH -c 20
#SBATCH --ntasks=1
#SBATCH -q gpu_normal
#SBATCH --time=1:00:00
#SBATCH --nice=1000
#SBATCH --gres=gpu:1
#SBATCH -C a100_80gb



# Environment setup
source $HOME/.bashrc
conda activate dinobloom2cond #dinobloom2cond #dinobloom
export PATH=$CONDA_PREFIX/bin:$PATH
cd /home/aih/manon.chossegros/from_sophia/dinov2/
#cd /home/haicu/sophia.wagner/projects/dinov2
#export CUDA_LAUNCH_BLOCKING=1
torchrun --nproc_per_node=1 dinov2/train/train.py --no-resume --config-file "dinov2/configs/train/custom_hier.yaml" --output_dir "models/dino_hier"
