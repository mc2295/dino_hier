# Hierarchical Cross-Entropy Enhances White Blood Cell Foundation Model Representations

## Overview
This repository extends the [DINOv2](https://github.com/facebookresearch/dinov2) foundation model by adding **supervised learning** through a classification head. We introduce **hierarchical cross-entropy (HXE)** to align predictions with biologically meaningful hierarchical structures, improving model interpretability and performance, particularly for rare classes.
This repository builds upon [DinoBloom](https://github.com/marrlab/DinoBloom) repository, introducing a hierarchical feature to improve organization and functionality.

## Datasets and Models
Datasets and training setting are the same as [DinoBloom](https://github.com/marrlab/DinoBloom) study. We start from DinoBloom-L pretrained model.

## Hierarchy

Design of the hierarchy. Classes are split by lineages first, and by maturity second. WBC labels available in the datasets are underlined.
![image1](https://github.com/user-attachments/assets/3c6ec692-73fa-4eb7-b95d-bc0e93c9e16e)

## Modified Loss function

Hierarchical cross entropy loss function is modified from its original version as to include non-leaf nodes in the possibly ground-truth classes.


![image](https://github.com/user-attachments/assets/bdc84e44-f805-47b9-98d9-8f2fb2aab717)

### Key Features:
- **Supervised Head Addition**: A classification head is integrated into **DINOv2**.
- **Cross-Entropy & Hierarchical Cross-Entropy Loss**: Standard **cross-entropy** and **HXE** loss are implemented for flexible training.
- **Hierarchical Labeling**: Enables training with multi-granular annotations from different datasets.

### Hierarchical Cross-Entropy:

Hierarchical Cross-Entropy is implemented according to the [HXE package](https://github.com/fiveai/making-better-mistakes/blob/master/README.md).


## Installation
```bash
git clone https://github.com/mc2295/dino_hier.git
conda env create -f environment.yml
```
## License

DINOv2 code and model weights are released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).


## Repository Structure : 

> **_NOTE:_** Here we present only the modifications from original DINOv2 repository 


```
├── dinov2
│   └── configs/
│      └── custom.yaml         # Configuration file with labels correspondances.
│   └── data/
│      └── datasets/
│          └── dataloader_sup.py/  # Introduction of supervised and unsupervised datasets  
│      └── augmentations.py         # Augmentation for Supervised images
│      └── loaders.py         # Make supervised and unsupervised dataloaders
│   └── loss/
│      └── hierarchical_ce.py   # Contains the tree structure and the hierarchical cross entropy
│      └── hierarchical_supcon.py   # Contains the hierarchical supcon loss
│
│   └── eval/
│      └── eval_extern.py   # Extract features and compute KNN and LogReg classification scores
│      └── eval_hier.py   # Evaluate model with hierarchical metrics
│      └── compute_metrics_hier.py   # Compute hierarchical metrics
│      └── compute_uncertainty.py   # Evaluate error intervals 
└── README.md                 # This file
|
└── train.sh        #Train model
```

---

---
## Usage
### Training the Model
```
sbatch train.sh
```
The user can adjust the following hyperparameters:


- **`supcon`**: `True` or `False`. Parameter to decide whether we use supcon loss (supcon hiersupcon) or not (CE, HierCE).
- **`hier`**: `True` or `False` Parameter to choose between flat loss or hierarchical supervision.
- **`version`**: (1, 2, 3) Determines which hierarchy to implement (`H1`, `H2`, `H3`).
- **`alpha`**: Sets up the alpha hyperparameter to weight edges in the hierarchical tree.

