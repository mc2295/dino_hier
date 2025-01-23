# SuperBloom – DinoBloom extension with supervised learning of cell labels

## Data overview

[Google sheet](https://docs.google.com/spreadsheets/d/184byqWGZ0Majwe1RrFV1GqKOBaoC4aUMiOf923IWaHM/edit?usp=sharing)

## Installation

Create conda env and install minimal requirements

```bash
conda create -n superbloom python=3.11
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda install numpy pandas h5py matplotlib wandb umap-learn transformers
pip install timm einops ftfy regex
```

```

```

## Evaluation

### Patch-level evaluation

```bash
export MODEL_PATH=/ictstr01/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitl_8_new_supcon/eval/training_4999
export MODEL_NAME=dinov2_vitl14
export EXP_NAME=superbloom_l

# acevedo
python dinov2/eval/patch_level/general_patch_eval.py --model_path $MODEL_PATH --model_name $MODEL_NAME --experiment_name $EXP_NAME --dataset_path /lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped

# apl cell
python dinov2/eval/patch_level/general_patch_eval.py --model_path $MODEL_PATH --dataset_path /ictstr01/groups/shared/histology_data/hematology_data/APL_AML --baseline --wandb apl_cell_eval --model_name $MODEL_NAME --experiment_name $EXP_NAME

# raabin
python dinov2/eval/patch_level/general_patch_eval.py --model_path $MODEL_PATH --model_name $MODEL_NAME --experiment_name $EXP_NAME --dataset_path /ictstr01/groups/shared/histology_data/hematology_data/raabin_wbc --baseline --wandb raabin_eval

# bone marrow
python dinov2/eval/patch_level/general_fixed_split_patch_eval.py --model_path $MODEL_PATH --model_name $MODEL_NAME --experiment_name $EXP_NAME --run_name bonemarrow --image_path_train /home/icb/valentin.koch/dinov2/dinov2/eval/patch_level/splits/bm_train.csv --image_path_test /home/icb/valentin.koch/dinov2/dinov2/eval/patch_level/splits/bm_test.csv

# rbc
python dinov2/eval/patch_level/general_fixed_split_patch_eval.py --model_path $MODEL_PATH --model_name $MODEL_NAME --experiment_name $EXP_NAME --run_name rbc_eval --image_path_test dinov2/eval/patch_level/splits/rbc_test.csv --image_path_train dinov2/eval/patch_level/splits/rbc_train.csv

# Cervix data $DATA_NAME = ['LBC', 'HiCervix', 'SIPaKMeD', 'ComparisonDetector']
python dinov2/eval/patch_level/general_fixed_split_patch_eval.py --model_path $MODEL_PATH --model_name $MODEL_NAME --experiment_name $EXP_NAME --run_name $DATA_NAME --image_path_test dinov2/eval/patch_level/splits/cerv_{$DATA_NAME}_test.csv --image_path_train dinov2/eval/patch_level/splits/cerv_{$DATA_NAME}_train.csv

```

### Patient-level evaluation

run `dinov2/scripts/eval_slide-level.sh`

or

```bash
python dinov2/eval/slide_level/eval_mil.py --dataset APL_AML_all --arch WBCMIL --model_name $MODEL_NAME --checkpoint /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8_new_supcon/eval/training_7999/teacher_checkpoint.pth --checkpoint_root /lustre/groups/shared/users/peng_marr/DinoBloomv2/superbloom/vitb_8_new_supcon 

```

## ToDos

* [ ] re-run all evals and collect commands for all

  * [ ] vits_reg
  * [ ] vits_supcon_no_koleo
  * [ ] vits_3M_350k_bs416_0.1ce+supcon_mlp_reg
  * [ ] vits_3M_350k_bs416_0.1ce+supcon_mlp
  * [ ] vitl_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc
  * [ ] vits_8
  * [ ] superbloom_vits_8
  * [ ] vits_2_8
  * [ ] vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc_a
  * [ ] DinoBloom_models
  * [ ] ctranspath
  * [ ] resnet50
  * [ ] resnet50_full
  * [ ] dinov2_vits14
  * [ ] dinov2_vitb14
  * [ ] dinov2_vitl14
  * [ ] dinov2_vitg14
  * [ ] owkin
  * [ ] uni
  * [ ] conch
  * [ ] superbloom
    * [ ] s 5999
    * [ ] b 6999
    * [ ] l 4999
    * [ ] g
  * [ ] vits_3M_350k_bs416_0.1ce+new_supcon_mlp_rbc 7999
  * [ ] vits_3M_350k_bs416_0.1ce+supcon_mlp_rbc 4999
* [ ] implement Transformer for MIL eval and check whether results are more consistent
