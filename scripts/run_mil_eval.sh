#!/bin/bash
source $HOME/.bashrc
cd /home/haicu/sophia.wagner/projects/dinov2
conda activate superbloom
python dinov2/eval/slide_level/eval_mil.py --dataset $1 --arch $2 --checkpoint $3 --model_name $4 --checkpoint_root $5

# test e.g. with
# bash scripts/run_MIL_eval.sh AML_Hehr WBCMIL /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon/eval/training_4999/teacher_checkpoint.pth dinov2_vitl14 /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon
# bash scripts/run_MIL_eval.sh APL_AML_all WBCMIL /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999/teacher_checkpoint.pth dinov2_vits14 /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8
# bash scripts/run_MIL_eval.sh Beluga WBCMIL /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999/teacher_checkpoint.pth dinov2_vits14 /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8
