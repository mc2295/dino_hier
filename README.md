# Hierarchical Cross-Entropy Enhances White Blood Cell Foundation Model Representations

## Overview
This repository extends the [DINOv2](https://github.com/facebookresearch/dinov2) foundation model by adding **supervised learning** through a classification head. We introduce **hierarchical cross-entropy (HCE)** to align predictions with biologically meaningful hierarchical structures, improving model interpretability and performance, particularly for rare classes.

## Datasets and Models
Datasets and training setting are the same as [DinoBloom](https://github.com/marrlab/DinoBloom) study. We start from DinoBloom-L pretrained model.

## Hierarchy

Design of the hierarchy. Classes are split by lineages first, and by maturity second. WBC labels available in the datasets are underlined.
![image1](https://github.com/user-attachments/assets/3c6ec692-73fa-4eb7-b95d-bc0e93c9e16e)


### Key Features:
- **Supervised Head Addition**: A classification head is integrated into **DINOv2**.
- **Cross-Entropy & Hierarchical Cross-Entropy Loss**: Standard **cross-entropy** and **HCE** loss are implemented for flexible training.
- **Hierarchical Labeling**: Enables training with multi-granular annotations from different datasets.


---
## Installation
```bash
git clone https://github.com/mc2295/dino_hier.git
cd dino_hier
pip install -r requirements.txt
```
## License

DINOv2 code and model weights are released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).


## Repository Structure : 

> **_NOTE:_** Here we present only the modifications from original DINOv2 repository 


```
├── dinov2
│   └── loss/
│      └── hierarchical_ce.py   # Contains the tree structure and the hierarchical cross entropy
│
│   └── config/
│      └── custom_hier.py         # Configuration file with labels correspondances
│
│   └── data/
│      └── datasets/
│          └── dataloader_sup.py/  # Introduction of supervised and unsupervised dataloaders  
│
│   └── eval/
│      └── eval_hier.py   # Evaluate model with non hierarchical and hierarchical metrics
└── README.md                 # This file
|
└── train_hier.sh        #Train model
```

---

---
## Usage
### Training the Model
```
sbatch train_hier.sh
```

