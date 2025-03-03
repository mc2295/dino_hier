#!/bin/bash

#SBATCH -o log/out_hier.txt
#SBATCH -e log/err_hier.txt
#SBATCH -J dino_hier
#SBATCH -p gpu_p
#SBATCH --ntasks=1
#SBATCH -q gpu_normal
#SBATCH --time=1:00:00
#SBATCH --nice=1000
#SBATCH --gres=gpu:1



# Environment setup
source $HOME/.bashrc
conda activate dinobloom2cond #dinobloom2cond #dinobloom
export PATH=$CONDA_PREFIX/bin:$PATH
#export CUDA_LAUNCH_BLOCKING=1
torchrun --nproc_per_node=1 dinov2_hier/train/train.py --no-resume --config-file "dinov2/configs/train/custom.yaml" --output_dir "models/dino_hier"
