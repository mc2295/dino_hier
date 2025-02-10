#!/bin/bash
# Define models in an array.
# Format for each element: 
# "description|MODEL_PATH|MODEL_ROOT|MODEL_NAME|EXP_NAME|CKPT_PATH"
MODELS=(
  # "superbloom L|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon/eval/training_4999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon|dinov2_vitl14|superbloom_vitl_8_new_supcon_4999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon/eval/training_4999/teacher_checkpoint.pth"
  
  # "superbloom S|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8|dinov2_vits14|superbloom_s_5999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999/teacher_checkpoint.pth"
  
  # "superbloom S (new supcon) 5999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon/eval/training_5999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon|dinov2_vits14|superbloom_s_new_supcon_5999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon/eval/training_5999/teacher_checkpoint.pth"
  
  # "superbloom S (new supcon) 7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon/eval/training_7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon|dinov2_vits14|superbloom_s_new_supcon_7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon/eval/training_7999/teacher_checkpoint.pth"
  
  # "superbloom B|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8/eval/training_6999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8|dinov2_vitb14|superbloom_vitb_8_6999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8/eval/training_6999/teacher_checkpoint.pth"
  
  # "superbloom B (new supcon)|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8_new_supcon/eval/training_7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8_new_supcon|dinov2_vitb14|superbloom_b_new_supcon_7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8_new_supcon/eval/training_7999/teacher_checkpoint.pth"
  
  # "vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc 4999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc/eval/training_4999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc|dinov2_vits14|vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc_4999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc/eval/training_4999/teacher_checkpoint.pth"
  
  # "vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc 7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc/eval/training_7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc|dinov2_vits14|vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc_7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc/eval/training_7999/teacher_checkpoint.pth"
  
  # "vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc 7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc/eval/training_7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc|dinov2_vits14|vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc_7999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc/eval/training_7999/teacher_checkpoint.pth"
  
  # "conch|/lustre/groups/shared/users/peng_marr/DinoBloomv2/conch/pytorch_model.bin|/lustre/groups/shared/users/peng_marr/DinoBloomv2/conch|conch|conch|/lustre/groups/shared/users/peng_marr/DinoBloomv2/conch/pytorch_model.bin"

  "conchv1.5|/lustre/groups/shared/users/peng_marr/DinoBloomv2/conchv1.5/conch_v1_5_pytorch_model.bin|/lustre/groups/shared/users/peng_marr/DinoBloomv2/conchv1.5|conchv1.5|conchv1.5|/lustre/groups/shared/users/peng_marr/DinoBloomv2/conchv1.5/conch_v1_5_pytorch_model.bin"

  # "uni|/lustre/groups/shared/users/peng_marr/DinoBloomv2/uni/pytorch_model.bin|/lustre/groups/shared/users/peng_marr/DinoBloomv2/uni|uni|uni|/lustre/groups/shared/users/peng_marr/DinoBloomv2/uni/pytorch_model.bin"

  "tong hempath|/lustre/groups/shared/users/peng_marr/DinoBloomv2/tong_hempath/eval/training_124999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/tong_hempath|dinov2_vitl16|tong_hempath_124999|/lustre/groups/shared/users/peng_marr/DinoBloomv2/tong_hempath/eval/training_124999/teacher_checkpoint.pth"
)
