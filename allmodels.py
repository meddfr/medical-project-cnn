import sys
import subprocess
import importlib
import os, random, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, recall_score, accuracy_score
from sklearn.preprocessing import label_binarize
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
import gc

warnings.filterwarnings('ignore')

SELECTED_MODEL = 'densenet121'

SEED = 42
IMG_SIZE = 300
LABEL_SMOOTHING = 0.1
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 25
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
WEIGHT_DECAY = 1e-2
CANCER_BOOST = 2.0
PNEUMONIA_BOOST = 3.0

BATCH_SIZES = {
    'resnet18': 64,
    'resnet50': 40,
    'densenet121': 32,
    'efficientnet_b3': 28,
}

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except (UnidentifiedImageError, OSError):
                index = (index + 1) % len(self)

def weighted_ce(logits, labels, weights):
    return F.cross_entropy(logits, labels, weight=weights, label_smoothing=LABEL_SMOOTHING)

def tversky_loss(logits, labels, num_classes, alpha=0.7, beta=0.3):
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(labels, num_classes).float()
    tp = (probs * onehot).sum(0)
    fp = (probs * (1 - onehot)).sum(0)
    fn = ((1 - probs) * onehot).sum(0)
    return (1 - tp / (tp + alpha * fp + beta * fn + 1e-8)).mean()

def tversky_confusion_loss(logits, labels, weights, cancer_idx, pneumonia_idx, num_classes, penalty=0.5):
    tv = tversky_loss(logits, labels, num_classes)
    probs = F.softmax(logits, dim=1)
    conf = 0.0
    c_mask = (labels == cancer_idx)
    p_mask = (labels == pneumonia_idx)
    if c_mask.sum() > 0:
        conf += probs[c_mask][:, pneumonia_idx].mean()
    if p_mask.sum() > 0:
        conf += probs[p_mask][:, cancer_idx].mean()
    return tv + penalty * conf

def predict_dual_threshold(model, loader, cancer_idx, pneumonia_idx, cancer_t, suppression_t=0.45):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            probs = F.softmax(model(imgs), dim=1).cpu().numpy()
            for p in probs:
                if p[cancer_idx] >= cancer_t and p[pneumonia_idx] < suppression_t:
                    y_pred.append(cancer_idx)
                else:
                    y_pred.append(p.argmax())
            y_true.extend(lbls.numpy())
    return np.array(y_true), np.array(y_pred)