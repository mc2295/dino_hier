#!/bin/bash
source $HOME/.bashrc
cd $7
conda activate $8
python dinov2/eval/patch_level/general_fixed_split_patch_eval.py \
    --model_path $1 \
    --model_name $2 \
    --experiment_name $3 \
    --run_name $4 \
    --image_path_train $5 \
    --image_path_test $6

# Example usage for testing:
# bash scripts/run_patch_eval_fixed.sh /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999 dinov2_vits14 superbloom_s rbc_superbloom_s dinov2/eval/patch_level/splits/rbc_train.csv dinov2/eval/patch_level/splits/rbc_test.csv /home/haicu/sophia.wagner/projects/dinov2 superbloom
# bash scripts/run_patch_eval_fixed.sh /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999 dinov2_vits14 superbloom_s rbc dinov2/eval/patch_level/splits/rbc_train.csv dinov2/eval/patch_level/splits/rbc_test.csv /home/haicu/sophia.wagner/projects/dinov2 superbloom
