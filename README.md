# DINOv2 with Hierarchical Supervision

## Overview
This repository extends the **DINOv2** foundation model by incorporating **supervised learning** through a classification head. We introduce **hierarchical cross-entropy (HCE)** to align predictions with biologically meaningful hierarchical structures, improving model interpretability and performance, particularly for rare classes.

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


