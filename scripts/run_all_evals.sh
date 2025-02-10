#!/bin/bash
set -e

# Source the model definitions
source scripts/model_infos.sh

# Define common paths and variables
LOG_DIR=/lustre/groups/shared/users/peng_marr/DinoBloomv2/job_scripts
PROJECT_PATH=/home/haicu/sophia.wagner/projects/dinov2
ENV_NAME=superbloom

# Function to submit jobs with common SLURM parameters
submit_job() {
    sbatch \
        --output="$LOG_DIR/${EXP_NAME}_log_%j.txt" \
        --error="$LOG_DIR/${EXP_NAME}_error_%j.txt" \
        --job-name="$EXP_NAME" \
        --partition=gpu_p \
        --qos=gpu_normal \
        --mem=240G \
        --time=8:00:00 \
        --nice=10000 \
        --gres=gpu:1 \
        "$@"
}

# Loop over each model and run all evaluations
for model_info in "${MODELS[@]}"; do
    # Split the model_info string using the '|' delimiter
    IFS="|" read -r model_desc MODEL_PATH MODEL_ROOT MODEL_NAME EXP_NAME CKPT_PATH <<< "$model_info"
    
    echo "Submitting evaluations for model: $model_desc ($EXP_NAME)"
    
    # === Patch Evaluation: Fixed splits ===
    
    # (a) bonemarrow split
    submit_job scripts/run_patch_eval_fixed.sh \
        "$MODEL_PATH" "$MODEL_ROOT" "$MODEL_NAME" "$EXP_NAME" "bonemarrow" \
        "dinov2/eval/patch_level/splits/bm_train.csv" \
        "dinov2/eval/patch_level/splits/bm_test.csv" \
        "$PROJECT_PATH" "$ENV_NAME"
    
    # (b) rbc_eval split
    submit_job scripts/run_patch_eval_fixed.sh \
        "$MODEL_PATH" "$MODEL_ROOT" "$MODEL_NAME" "$EXP_NAME" "rbc_eval" \
        "dinov2/eval/patch_level/splits/rbc_train.csv" \
        "dinov2/eval/patch_level/splits/rbc_test.csv" \
        "$PROJECT_PATH" "$ENV_NAME"
    
    # === Patch Evaluation: Cross-Validation ===
    
    # (a) Acevedo
    submit_job scripts/run_patch_eval.sh \
        "$MODEL_PATH" "$MODEL_ROOT" "$MODEL_NAME" "dino_eval" \
        "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped" \
        "dino_eval" "$PROJECT_PATH" "$ENV_NAME"
    
    # (b) APL cell
    submit_job scripts/run_patch_eval.sh \
        "$MODEL_PATH" "$MODEL_ROOT" "$MODEL_NAME" "apl_cell_eval" \
        "/lustre/groups/shared/histology_data/hematology_data/APL_AML" \
        "apl_cell_eval" "$PROJECT_PATH" "$ENV_NAME"
    
    # (c) Raabin
    submit_job scripts/run_patch_eval.sh \
        "$MODEL_PATH" "$MODEL_ROOT" "$MODEL_NAME" "raabin_eval" \
        "/lustre/groups/shared/histology_data/hematology_data/raabin_wbc" \
        "raabin_eval" "$PROJECT_PATH" "$ENV_NAME"
    
    # === Slide Evaluation (MIL) ===
    
    submit_job scripts/run_mil_eval.sh "APL_AML_all" "WBCMIL" "$CKPT_PATH" "$MODEL_NAME" "$MODEL_ROOT"
    submit_job scripts/run_mil_eval.sh "AML_Hehr" "WBCMIL" "$CKPT_PATH" "$MODEL_NAME" "$MODEL_ROOT"
    submit_job scripts/run_mil_eval.sh "Beluga" "WBCMIL" "$CKPT_PATH" "$MODEL_NAME" "$MODEL_ROOT"
    
done
