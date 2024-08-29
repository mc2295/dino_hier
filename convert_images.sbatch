#!/bin/bash

#SBATCH -o /home/haicu/sophia.wagner/slurm/logs/dinov2/std_out_%j.job
#SBATCH -e /home/haicu/sophia.wagner/slurm/logs/dinov2/std_err_%j.job
#SBATCH -J dino
#SBATCH -p gpu_p
#SBATCH -c 20
#SBATCH --mem=160G
#SBATCH -q gpu_normal
#SBATCH --time=48:00:00
#SBATCH --nice=1000
#SBATCH --gres=gpu:0

# Environment setup
source $HOME/.bashrc
conda activate dinobloom

cd /home/haicu/sophia.wagner/projects/dinov2

python convert_images.py
