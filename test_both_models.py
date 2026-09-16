"""
Compare both segmentation models on the SeabedObjects classification task.
Extracts ResNet34 encoders from each trained segmentation checkpoint,
adds a classification head, and evaluates on ship vs airplane.
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ── Dataset ──────────────────────────────────────────────────────────────
class SeabedDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        img = F.interpolate(img.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)[0]
        label = self.labels[idx]
        return img, label


def load_seabed_data():
    root = Path(r"D:\SIH\SeabedObjects-Ship-and-Airplane-dataset")
    image_paths = []
    labels = []

    # Ships (label=0)
    for folder in ["ship-real-1", "ship-real-2", "ship-real-3"]:
        folder_path = root / folder
        if folder_path.exists():
            for f in sorted(folder_path.iterdir()):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    image_paths.append(str(f))
                    labels.append(0)

    # Planes (label=1)
    plane_dir = root / "plane-real" / "plane-real"
    if plane_dir.exists():
        for f in sorted(plane_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                image_paths.append(str(f))
                labels.append(1)

    return image_paths, labels


def stratified_split(image_paths, labels, val_frac=0.15, test_frac=0.15, seed=42):
    """Stratified split by class."""
    rng = random.Random(seed)
    by_class = {}
    for i, (p, l) in enumerate(zip(image_paths, labels)):
        by_class.setdefault(l, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for cls, indices in by_class.items():
        rng.shuffle(indices)
        n_test = int(len(indices) * test_frac)
        n_val = int(len(indices) * val_frac)
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


# ── Model A: AI4Shipwrecks encoder (1ch input, scSE attention) ──────────
def load_model_a_encoder():
    """Load ResNet34 encoder from AI4Shipwrecks best_model_v1_iou0.71.pth"""
    import segmentation_models_pytorch as smp

    ckpt_path = r"D:\SIH\AI4Shipwrecks\AI4Shipwrecks\runs\best_model_v1_iou0.71.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Build a fresh 1ch encoder and load only encoder.* keys
    encoder = smp.encoders.get_encoder("resnet34", in_channels=1, depth=5, weights=None)
    encoder_state = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
    result = encoder.load_state_dict(encoder_state, strict=False)
    print(f"[Model A] Loaded encoder from {ckpt_path}")
    print(f"[Model A] Checkpoint val_iou: {ckpt.get('val_iou', 'N/A')}")
    if result and result.missing_keys:
        print(f"[Model A] Missing keys: {len(result.missing_keys)}")
    return encoder


# ── Model B: AI4Shipwrecks_Project encoder (3ch input) ───────────────────
def load_model_b_encoder():
    """Load ResNet34 encoder from AI4Shipwrecks_Project baseline checkpoint."""
    import segmentation_models_pytorch as smp

    ckpt_path = r"D:\SIH\AI4Shipwrecks_Project\checkpoints\baseline_unet_resnet34_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Build a fresh 3ch encoder and load only encoder.* keys
    encoder = smp.encoders.get_encoder("resnet34", in_channels=3, depth=5, weights=None)
    encoder_state = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
    result = encoder.load_state_dict(encoder_state, strict=False)
    print(f"[Model B] Loaded encoder from {ckpt_path}")
    print(f"[Model B] Checkpoint val_dice: {ckpt.get('best_val_dice', 'N/A')}")
    if result and result.missing_keys:
        print(f"[Model B] Missing keys: {len(result.missing_keys)}")
    return encoder


# ── Classifier head ──────────────────────────────────────────────────────
class ClassifierHead(nn.Module):
    def __init__(self, encoder, num_classes=2, replicate_channels=1):
        super().__init__()
        self.encoder = encoder
        self.replicate_channels = replicate_channels
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        if self.replicate_channels > 1:
            x = x.repeat(1, self.replicate_channels, 1, 1)
        feats = self.encoder(x)[-1]  # last feature map
        return self.head(feats)


# ── Train classifier ─────────────────────────────────────────────────────
def train_classifier(encoder, train_loader, val_loader, name, epochs=20, lr=1e-3, replicate_channels=1):
    set_seed(SEED)
    model = ClassifierHead(encoder, num_classes=2, replicate_channels=replicate_channels).to(DEVICE)

    # Weighted CE for class imbalance
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  [{name}] Epoch {epoch:2d} | loss={total_loss/total:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"  [{name}] Best val accuracy: {best_val_acc:.4f}")
    return model


# ── Evaluate and plot confusion matrix ───────────────────────────────────
def evaluate_and_plot(model, test_loader, class_names, save_path, model_name):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Test Results")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\n{report}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{model_name}\nAccuracy: {acc:.4f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Confusion matrix saved to {save_path}")
    plt.close()

    return acc, cm


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)

    print("Loading SeabedObjects dataset...")
    image_paths, labels = load_seabed_data()
    print(f"  Total: {len(image_paths)} images ({sum(l==0 for l in labels)} ships, {sum(l==1 for l in labels)} planes)")

    train_idx, val_idx, test_idx = stratified_split(image_paths, labels)
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    full_dataset = SeabedDataset(image_paths, labels)
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    test_subset = torch.utils.data.Subset(full_dataset, test_idx)

    common = dict(num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, **common)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, **common)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, **common)

    results = {}

    # ── Model A ──
    print("\n" + "="*60)
    print("  MODEL A: AI4Shipwrecks (UNet+ResNet34+scSE, 1ch input)")
    print("="*60)
    encoder_a = load_model_a_encoder()
    model_a = train_classifier(encoder_a, train_loader, val_loader, "Model A", epochs=20)
    acc_a, cm_a = evaluate_and_plot(
        model_a, test_loader,
        ["Ship", "Airplane"],
        r"D:\SIH\confusion_matrix_model_a.png",
        "Model A: AI4Shipwrecks (UNet+ResNet34+scSE)"
    )
    results["A"] = acc_a

    # ── Model B ──
    print("\n" + "="*60)
    print("  MODEL B: AI4Shipwrecks_Project (UNet+ResNet34, 3ch input)")
    print("="*60)
    encoder_b = load_model_b_encoder()
    model_b = train_classifier(encoder_b, train_loader, val_loader, "Model B", epochs=20, replicate_channels=3)
    acc_b, cm_b = evaluate_and_plot(
        model_b, test_loader,
        ["Ship", "Airplane"],
        r"D:\SIH\confusion_matrix_model_b.png",
        "Model B: AI4Shipwrecks_Project (UNet+ResNet34)"
    )
    results["B"] = acc_b

    # ── Summary ──
    print("\n" + "="*60)
    print("  FINAL COMPARISON")
    print("="*60)
    print(f"  Model A (AI4Shipwrecks)  — Test Accuracy: {results['A']:.4f}")
    print(f"  Model B (AI4Shipwrecks_Project) — Test Accuracy: {results['B']:.4f}")
    winner = "A" if results["A"] > results["B"] else "B"
    print(f"\n  Winner: Model {winner}")
    print(f"\n  Confusion matrices saved to:")
    print(f"    D:\\SIH\\confusion_matrix_model_a.png")
    print(f"    D:\\SIH\\confusion_matrix_model_b.png")


if __name__ == "__main__":
    main()
