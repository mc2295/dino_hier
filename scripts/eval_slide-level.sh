
# # baselines
# python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name resnet50      --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/resnet50
# python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name resnet50_full --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/resnet50_full
# python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vits14 --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/dinov2_vits14
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vitb14 --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/dinov2_vitb14
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vitl14 --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/dinov2_vitl14
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vitg14 --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/dinov2_vitg14
## # histo baselines
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name retccl        --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/retccl
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name ctranspath    --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/ctranspath
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name owkin         --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/owkin
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name uni           --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/uni
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name conch         --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/conch
# # dinobloom 
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-S.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-S
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vitb14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-B.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-B
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vitl14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-L.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-L
#python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vitg14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-G.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-G
# # superbloom 
python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon/eval/training_7999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon 
python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8
python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc/eval/training_7999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc
python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc/eval/training_7999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc

python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vitb14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8_new_supcon/eval/training_7999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8_new_supcon 
python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name dinov2_vitl14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon/eval/training_4999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon 


# # baselines
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name resnet50      --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/resnet50
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name resnet50_full --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/resnet50_full
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vits14 --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/dinov2_vits14
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vitb14 --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/dinov2_vitb14
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vitl14 --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/dinov2_vitl14
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vitg14 --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/dinov2_vitg14
# # histo baselines
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name retccl        --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/retccl
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name ctranspath    --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/ctranspath
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name owkin         --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/owkin
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name uni           --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/uni
#python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name conch         --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/conch
# # dinobloom 
python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-S.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-S
python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vitb14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-B.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-B
python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vitl14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-L.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-L
python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vitg14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-G.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/DinoBloom_models/DinoBloom-G
# # superbloom 
python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon/eval/training_7999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8_new_supcon 
python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8/eval/training_5999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vits_8
python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc/eval/training_7999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc
python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vits14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc/eval/training_7999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc

python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vitb14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8_new_supcon/eval/training_7999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8_new_supcon 
python dinov2/eval/slide_level/eval_mil.py --dataset AML_Hehr --arch WBCMIL --model_name dinov2_vitl14 --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon/eval/training_4999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon 
