#!/bin/bash

# Define common paths and variables
MODEL_PATH=/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc/eval/training_7999
CKPT_PATH=/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc/eval/training_7999/teacher_checkpoint.pth
MODEL_ROOT=/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc
MODEL_NAME=dinov2_vits14
EXP_NAME=vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc

LOG_DIR=/lustre/groups/shared/users/peng_marr/DinoBloomv2/job_scripts
PROJECT_PATH=/home/haicu/sophia.wagner/projects/dinov2
ENV_NAME=superbloom

# Function to submit jobs with common SLURM parameters
submit_job() {    
    sbatch \
        --output=$LOG_DIR/${EXP_NAME}_log_%j.txt \
        --error=$LOG_DIR/${EXP_NAME}_error_%j.txt \
        --job-name=$EXP_NAME \
        --partition=gpu_p \
        --qos=gpu_normal \
        --time=8:00:00 \
        --nice=10000 \
        --gres=gpu:1 \
        "$@"
}

# Run patch eval with fixed splits
submit_job scripts/run_patch_eval_fixed.sh \
    "$MODEL_PATH" "$MODEL_NAME" "$EXP_NAME" "bonemarrow" \
    "dinov2/eval/patch_level/splits/bm_train.csv" \
    "dinov2/eval/patch_level/splits/bm_test.csv" \
    "$PROJECT_PATH" "$ENV_NAME"

submit_job scripts/run_patch_eval_fixed.sh \
    "$MODEL_PATH" "$MODEL_NAME" "$EXP_NAME" "rbc_eval" \
    "dinov2/eval/patch_level/splits/rbc_train.csv" \
    "dinov2/eval/patch_level/splits/rbc_test.csv" \
    "$PROJECT_PATH" "$ENV_NAME"

# Run patch eval with cross-validation
submit_job scripts/run_patch_eval.sh \
    "$MODEL_PATH" "$MODEL_NAME" "$EXP_NAME" \
    "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped" \
    "dino_eval" "$PROJECT_PATH" "$ENV_NAME"

submit_job scripts/run_patch_eval.sh \
    "$MODEL_PATH" "$MODEL_NAME" "$EXP_NAME" \
    "/lustre/groups/shared/histology_data/hematology_data/APL_AML" \
    "apl_cell_eval" "$PROJECT_PATH" "$ENV_NAME"

submit_job scripts/run_patch_eval.sh \
    "$MODEL_PATH" "$MODEL_NAME" "$EXP_NAME" \
    "/lustre/groups/shared/histology_data/hematology_data/raabin_wbc" \
    "raabin_eval" "$PROJECT_PATH" "$ENV_NAME"

# Run slide eval
submit_job scripts/run_mil_eval.sh "APL_AML_all" "WBCMIL" "$CKPT_PATH" "$MODEL_NAME" "$MODEL_ROOT"

submit_job scripts/run_mil_eval.sh "AML_Hehr" "WBCMIL" "$CKPT_PATH" "$MODEL_NAME" "$MODEL_ROOT"

submit_job scripts/run_mil_eval.sh "Beluga" "WBCMIL" "$CKPT_PATH" "$MODEL_NAME" "$MODEL_ROOT"
