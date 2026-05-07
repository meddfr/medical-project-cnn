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
from torchvision.datasets import ImageFolder
from PIL import UnidentifiedImageError
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
NON_INFECTIOUS = ['lung cancer', 'lungcancer', 'cancer']

BATCH_SIZES = {
    'resnet18': 64,
    'resnet50': 40,
    'densenet121': 32,
    'efficientnet_b3': 28,
}

BASE_PATH = '/kaggle/input/datasets/mohamedfarhi31/medical-project/Medical_Project/organized_dataset'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')
VAL_PATH = os.path.join(BASE_PATH, 'val')
TEST_PATH = os.path.join(BASE_PATH, 'test')
SAVE_DIR = '/kaggle/working/results'
os.makedirs(SAVE_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    gc.collect()
else:
    device = torch.device('cpu')

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

def build_loaders(batch_size):
    set_seed()
    train_ds = SafeImageFolder(TRAIN_PATH, train_transforms)
    val_ds = SafeImageFolder(VAL_PATH, val_transforms)
    test_ds = SafeImageFolder(TEST_PATH, val_transforms)
    label_counts = Counter(train_ds.targets)
    kw = dict(num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader, train_ds.classes, label_counts

def compute_weights(class_names, label_counts):
    n, total = len(class_names), sum(label_counts.values())
    w = torch.tensor([total / (n * label_counts[i]) for i in range(n)], dtype=torch.float)
    for i, name in enumerate(class_names):
        if 'cancer' in name.lower():
            w[i] *= CANCER_BOOST
        if 'pneumonia' in name.lower():
            w[i] *= PNEUMONIA_BOOST
    return w.to(device)

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

def get_model(arch, num_classes):
    set_seed()
    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=True)
        model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    model = model.to(device)
    return model

def train_epoch(model, loader, loss_fn, optimizer, scaler, desc):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for imgs, lbls in pbar:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            loss = loss_fn(model(imgs), lbls)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

def find_threshold(model, val_loader, cancer_idx):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader, desc="Finding threshold", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            with autocast('cuda'):
                probs = F.softmax(model(imgs), dim=1).cpu().float()
            all_probs.append(probs)
            all_true.extend(lbls.numpy())
    probs_np = torch.cat(all_probs).numpy()
    binary_true = (np.array(all_true) == cancer_idx).astype(int)
    prec, rec, thresholds = precision_recall_curve(binary_true, probs_np[:, cancer_idx])
    f1 = np.where((prec[:-1] + rec[:-1]) == 0, 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]))
    best_t = float(thresholds[np.argmax(f1)])
    return best_t

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

def to_binary(labels, class_names):
    return np.array([0 if any(c in class_names[l].lower() for c in NON_INFECTIOUS) else 1 for l in labels])

def save_confusion_matrix(y_true, y_pred, class_names, arch, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=f'{arch.upper()} - Confusion Matrix',
           ylabel='True Label', xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{arch}_confusion_matrix.png'), dpi=150)
    plt.close()

def save_training_plots(history, arch, save_dir):
    val_history = [h for h in history if h['val_acc'] > 0]
    if not val_history:
        return
    df = pd.DataFrame(val_history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    all_df = pd.DataFrame(history)
    axes[0].plot(all_df['epoch'], all_df['loss'], 'b-', alpha=0.5, label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{arch.upper()} - Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df['epoch'], df['val_acc'], 'g-', linewidth=2, label='Validation Accuracy')
    axes[1].plot(df['epoch'], df['cancer_recall'], 'r-', linewidth=2, label='Cancer Recall')
    axes[1].plot(df['epoch'], df['pneumonia_recall'], 'orange', linewidth=2, label='Pneumonia Recall')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'{arch.upper()} - Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{arch}_training_history.png'), dpi=150)
    plt.close()

def train_model(arch):
    torch.cuda.empty_cache()
    gc.collect()
    batch_size = BATCH_SIZES[arch]
    train_loader, val_loader, test_loader, class_names, label_counts = build_loaders(batch_size)
    cancer_idx = next(i for i, n in enumerate(class_names) if 'cancer' in n.lower())
    pneumonia_idx = next(i for i, n in enumerate(class_names) if 'pneumonia' in n.lower())
    num_classes = len(class_names)
    weights = compute_weights(class_names, label_counts)
    model = get_model(arch, num_classes)
    scaler = GradScaler('cuda')
    start_time = time.time()
    history = []
    epoch_counter = 0
    optimizer_p1 = optim.AdamW(model.parameters(), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)
    loss_fn_p1 = lambda logits, lbls: weighted_ce(logits, lbls, weights)
    for epoch in range(EPOCHS_PHASE1):
        loss = train_epoch(model, train_loader, loss_fn_p1, optimizer_p1, scaler, f"Warmup {epoch+1}/{EPOCHS_PHASE1}")
        epoch_counter += 1
        history.append({'epoch': epoch_counter, 'loss': loss, 'val_acc': 0, 'cancer_recall': 0, 'pneumonia_recall': 0})
        torch.cuda.empty_cache()
    optimizer_p2 = optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer_p2, T_max=EPOCHS_PHASE2, eta_min=1e-6)
    loss_fn_p2 = lambda logits, lbls: tversky_confusion_loss(logits, lbls, weights, cancer_idx, pneumonia_idx, num_classes)
    best_combined = 0.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(EPOCHS_PHASE2):
        loss = train_epoch(model, train_loader, loss_fn_p2, optimizer_p2, scaler, f"Main {epoch+1}/{EPOCHS_PHASE2}")
        scheduler.step()
        epoch_counter += 1
        if (epoch + 1) % 3 == 0 or epoch == EPOCHS_PHASE2 - 1:
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for imgs, lbls in tqdm(val_loader, desc="Validating", leave=False):
                    imgs = imgs.to(device, non_blocking=True)
                    with autocast('cuda'):
                        probs = F.softmax(model(imgs), dim=1).cpu().float()
                    all_probs.append(probs)
                    all_labels.extend(lbls.numpy())
            probs_np = torch.cat(all_probs).numpy()
            y_true_val = np.array(all_labels)
            binary_true = (y_true_val == cancer_idx).astype(int)
            prec, rec, thresholds = precision_recall_curve(binary_true, probs_np[:, cancer_idx])
            f1_scores = np.where((prec[:-1] + rec[:-1]) == 0, 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]))
            current_threshold = float(thresholds[np.argmax(f1_scores)]) if len(thresholds) > 0 else 0.3
            y_pred_val = []
            for p in probs_np:
                if p[cancer_idx] >= current_threshold and p[pneumonia_idx] < 0.45:
                    y_pred_val.append(cancer_idx)
                else:
                    y_pred_val.append(p.argmax())
            cancer_recall = recall_score(y_true_val, y_pred_val, labels=[cancer_idx], average=None, zero_division=0)[0]
            pneumonia_recall = recall_score(y_true_val, y_pred_val, labels=[pneumonia_idx], average=None, zero_division=0)[0]
            val_acc = accuracy_score(y_true_val, y_pred_val)
            combined = 0.6 * cancer_recall + 0.4 * pneumonia_recall
            history.append({'epoch': epoch_counter, 'loss': loss, 'val_acc': val_acc, 'cancer_recall': cancer_recall, 'pneumonia_recall': pneumonia_recall})
            if combined >= best_combined:
                best_combined = combined
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            history.append({'epoch': epoch_counter, 'loss': loss, 'val_acc': 0, 'cancer_recall': 0, 'pneumonia_recall': 0})
        torch.cuda.empty_cache()
    model.load_state_dict(best_state)
    training_time = round((time.time() - start_time) / 60, 1)
    save_training_plots(history, arch, SAVE_DIR)
    cancer_threshold = find_threshold(model, val_loader, cancer_idx)
    y_true, y_pred = predict_dual_threshold(model, test_loader, cancer_idx, pneumonia_idx, cancer_threshold)
    save_confusion_matrix(y_true, y_pred, class_names, arch, SAVE_DIR)
    binary_true = to_binary(y_true, class_names)
    binary_pred = to_binary(y_pred, class_names)
    results = {
        'model': arch,
        'weights': 'torchvision_pretrained',
        'batch_size': batch_size,
        'image_size': IMG_SIZE,
        'epochs': EPOCHS_PHASE1 + EPOCHS_PHASE2,
        'accuracy': accuracy_score(y_true, y_pred),
        'cancer_recall': recall_score(y_true, y_pred, labels=[cancer_idx], average=None, zero_division=0)[0],
        'pneumonia_recall': recall_score(y_true, y_pred, labels=[pneumonia_idx], average=None, zero_division=0)[0],
        'training_time_min': training_time,
        'best_combined_score': best_combined,
        'cancer_threshold': cancer_threshold
    }
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{arch}_torchvision_pretrained.pth'))
    pd.DataFrame([results]).to_csv(os.path.join(SAVE_DIR, f'{arch}_results.csv'), index=False)
    del model, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    return results

if __name__ == '__main__':
    set_seed()
    result = train_model(SELECTED_MODEL)
