# Dataset Sources

This project uses chest X-ray images from the following public Kaggle datasets.

## Sources

| Pathology | Source | Link |
|-----------|--------|------|
| COVID-19 | COVID-19 Image Dataset by Pranav Raikote | https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset |
| Lung Cancer | IQ-OTH/NCCD Lung Cancer Dataset | https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xray-database |
| Pneumonia | Chest X-Ray Images (Pneumonia) | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| Tuberculosis | Tuberculosis Chest X-ray Database | https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xray-database |

## Important Note About COVID-19 Dataset

The COVID-19 dataset contains three classes: COVID-19, Viral Pneumonia, and Normal. For this project, only COVID-19 images were extracted.

## Dataset Organization

The images were organized into the following structure:
organized_dataset/
├── train/
│ ├── covid/
│ ├── lung_cancer/
│ ├── pneumonia/
│ └── tuberculosis/
├── val/
│ ├── covid/
│ ├── lung_cancer/
│ ├── pneumonia/
│ └── tuberculosis/
└── test/
├── covid/
├── lung_cancer/
├── pneumonia/
└── tuberculosis/

text

## How to Reproduce

1. Download each dataset from the links above
2. Extract all images
3. Create the folder structure above
4. Copy images into corresponding folders
5. Split images into train/val/test (approximately 70/15/15)
6. Update `BASE_PATH` in `train_model.py` to point to your `organized_dataset` folder

## License

These datasets are publicly available on Kaggle. Please refer to each dataset's page for specific license terms.
