#!/bin/bash
source $HOME/.bashrc
cd $7
conda activate $8
python dinov2/eval/patch_level/general_patch_eval.py --model_path $1 --checkpoint_root $2 --model_name $3 --experiment_name $4 --dataset_path $5 --wandb $6

# test e.g. with
# bash scripts/run_patch_eval.sh /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999 dinov2_vits14 superbloom_s /lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped dino_eval /home/haicu/sophia.wagner/projects/dinov2 superbloom
# bash scripts/run_patch_eval.sh /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999 dinov2_vits14 superbloom_s /lustres/groups/shared/histology_data/hematology_data/APL_AML apl_cell_eval /home/haicu/sophia.wagner/projects/dinov2 superbloom
# bash scripts/run_patch_eval.sh /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999 dinov2_vits14 superbloom_s /lustre/groups/shared/histology_data/hematology_data/raabin_wbc raabin_eval /home/haicu/sophia.wagner/projects/dinov2 superbloom