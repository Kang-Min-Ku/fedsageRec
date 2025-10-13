# Federated Recommender System with Data Valuation for E-commerce Platform

This repository contains the official implementation of the paper *"Federated Recommender System with Data Valuation for E-commerce Platform,"* published in *Expert Systems with Applications (ESWA).*  
[Paper (ESWA)](https://www.sciencedirect.com/science/article/pii/S095741742503310X) | [Preprint (arXiv)](https://www.arxiv.org/pdf/2509.11196)

# Installation

1. **Clone repository**
```bash
git clone https://github.com/Kang-Min-Ku/FedGDVE.git
```
2. **Download environment**

    Download the pre-configured environment from the following link: 📦[Drive](https://drive.google.com/file/d/1OD0lcIxj49mdtt9ffQwd1cszo8qQUajI/view?usp=drive_link)

3. **Download FedML module**

    Since we modified a very old version of [FedML](https://github.com/FedML-AI/FedML), it needs to be downloaded from the following link manually: 📦[Drive](https://drive.google.com/file/d/1QQF2BQp9FNwWcQGWo-H-HHZcZPd2o6Ma/view?usp=drive_link)

4. **Place FedML module in the root directory of this repository.**

# How to Run

1. **Activate environment**
   ```bash
   conda activate [absolute path of env]
   ```
2. **Change directory**
    ```bash
    cd experiments/distributed/recommendation
    ```
3. **Run code**
   ```bash
   sh run_fedgdve.sh [number of worker] [config file path]
   ```
   The configuration file is located in the ```config``` directory as a ```.yaml``` file.

# Citation

```bibtex
@article{kang2025federated,
  title={Federated recommender system with data valuation for e-commerce platform},
  author={Park, Jongwon and Kang, Minku and Sim, Wooseok and Lee, Soyoung and Park, Hogun},
  journal={Expert Systems with Applications},
  year={2025},
}
```