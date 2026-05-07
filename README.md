# Chest X-Ray Classification: COVID-19, Lung Cancer, Pneumonia, Tuberculosis

This project compares four CNN architectures (ResNet18, ResNet50, DenseNet121, EfficientNet-B3) for classifying chest X-rays into four pathologies. It also provides a binary classification (infectious vs non-infectious) for clinical triage.

## Features

- Four-class classification (COVID-19, lung cancer, pneumonia, tuberculosis)
- Binary classification (infectious vs non-infectious)
- Two-phase training (warmup + fine-tuning)
- Tversky loss with confusion penalty targeting pneumonia-cancer misclassifications
- Dual-threshold prediction strategy
- Per-class weights to handle class imbalance

## Models Compared

| Model | Parameters | Batch Size |
|-------|------------|------------|
| ResNet18 | 11M | 64 |
| ResNet50 | 25M | 40 |
| DenseNet121 | 8M | 32 |
| EfficientNet-B3 | 12M | 28 |

## Dataset

The dataset consists of chest X-ray images from four public Kaggle sources:

| Pathology | Source |
|-----------|--------|
| COVID-19 | COVID-19 Image Dataset |
| Lung Cancer | IQ-OTH/NCCD Lung Cancer Dataset |
| Pneumonia | Chest X-Ray Images (Pneumonia) |
| Tuberculosis | Tuberculosis Chest X-ray Database |

See [DATA_SOURCES.md](DATA_SOURCES.md) for full details and links.

## Requirements

- Python 3.10
- PyTorch 2.0+
- torchvision 0.15+
- numpy, pandas, matplotlib, seaborn
- scikit-learn, tqdm, PIL

## Installation

```bash
git clone https://github.com/meddfr/medical-project-cnn.git
cd medical-project-cnn
pip install -r requirements.txt
