#!/bin/bash

base_dir="/lustre/groups/shared/users/peng_marr/DinoBloomv2/hematology_patches_separated/"
output_dir="/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc/train_embeddings"
model_name="dinov2_vits14"
checkpoint="/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc/eval/training_4999/teacher_checkpoint.pth"
#!/bin/bash

job_script_dir="/lustre/groups/shared/users/peng_marr/DinoBloomv2/job_scripts"

# Create the job script directory if it doesn't exist
mkdir -p "$job_script_dir"

# Find all .txt files in the base directory and subdirectories
find "$base_dir" -type f -name "*.txt" | while IFS= read -r txt_file; do
    # Create a unique job name based on the txt file name
    job_name=$(basename "$txt_file" .txt)
    
    # Create a SLURM job script
    job_script="$job_script_dir/${job_name}.sbatch"
    cat <<EOT > "$job_script"
#!/bin/bash

#SBATCH --job-name=${job_name}
#SBATCH --output=${job_script_dir}/${job_name}_%j.out
#SBATCH --error=${job_script_dir}/${job_name}_%j.err
#SBATCH -p gpu_p
#SBATCH -q gpu_normal
#SBATCH -c 20
#SBATCH --mem=150G
#SBATCH --time=2-0:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1

source $HOME/.bashrc
conda activate dinobloom
cd /home/haicu/sophia.wagner/projects/dinov2

python dinov2/eval/extract_features.py "$txt_file" "$output_dir" "$model_name" "$checkpoint"

EOT

    # Submit the job script using sbatch
    sbatch "$job_script"
done

