import random
from collections import Counter
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_ROOT = Path(r"D:\SIH\SeabedObjects-Ship-and-Airplane-dataset")
WRECK_CKPT = Path(__file__).resolve().parent / "runs" / "best_model_v1_iou0.71.pth"
OUT_DIR = Path(__file__).resolve().parent / "runs"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
HEAD_LR = 1e-3
ENCODER_LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples():
    samples = []
    for sub in ["ship-real-1", "ship-real-2", "ship-real-3"]:
        for p in (DATA_ROOT / sub).iterdir():
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                samples.append((p, 0))
    plane_dir = DATA_ROOT / "plane-real" / "plane-real"
    for p in plane_dir.iterdir():
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            samples.append((p, 1))
    return samples


def load_image(path):
    img = Image.open(path).convert("L")
    return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)[None]


class SonarObjectsDataset(Dataset):
    def __init__(self, samples, train_mode):
        self.samples = samples
        self.train_mode = train_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_image(path)
        _, h, w = img.shape
        if self.train_mode:
            scale = random.uniform(0.6, 1.0)
            ch = min(h, max(32, int(h * scale)))
            cw = min(w, max(32, int(w * scale)))
            y0 = random.randint(0, h - ch)
            x0 = random.randint(0, w - cw)
            img = img[:, y0 : y0 + ch, x0 : x0 + cw]
        img = F.interpolate(
            img[None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
            align_corners=False, antialias=True,
        )[0]
        if self.train_mode:
            if random.random() < 0.5:
                img = torch.flip(img, [-1])
            if random.random() < 0.5:
                img = torch.flip(img, [-2])
            k = random.randint(0, 3)
            img = torch.rot90(img, k, [-2, -1])
            img = img * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)
            img = img.clamp(0.0, 1.0) + torch.randn_like(img) * 0.02
        return img.clamp(0.0, 1.0), label


def stratified_split(samples, seed=SEED):
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for s in samples:
        by_class[s[1]].append(s)
    train, val, test = [], [], []
    for cls, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(n * 0.15))
        n_test = max(1, int(n * 0.15))
        test += items[:n_test]
        val += items[n_test : n_test + n_val]
        train += items[n_test + n_val :]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class SonarClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        feats = self.encoder(x)[-1]
        return self.head(feats)


def build_classifier(pretrain):
    weights = "imagenet" if pretrain == "imagenet" else None
    encoder = smp.encoders.get_encoder("resnet34", in_channels=1, depth=5, weights=weights)
    if pretrain == "wreck":
        ckpt = torch.load(WRECK_CKPT, map_location="cpu", weights_only=True)
        enc_sd = {
            k[len("encoder."):]: v
            for k, v in ckpt["model_state_dict"].items()
            if k.startswith("encoder.")
        }
        encoder.load_state_dict(enc_sd, strict=True)
        print(f"Loaded shipwreck encoder weights from {WRECK_CKPT.name} ({len(enc_sd)} tensors)")
    model = SonarClassifier(encoder)
    nn.init.zeros_(model.head[-1].bias)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for cls in (0, 1):
            tp[cls] += ((preds == cls) & (labels == cls)).sum().item()
            fp[cls] += ((preds == cls) & (labels != cls)).sum().item()
            fn[cls] += ((preds != cls) & (labels == cls)).sum().item()
    acc = correct / max(total, 1)
    f1s = []
    for cls in (0, 1):
        prec = tp[cls] / max(tp[cls] + fp[cls], 1)
        rec = tp[cls] / max(tp[cls] + fn[cls], 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-7))
    macro_f1 = sum(f1s) / len(f1s)
    return acc, macro_f1, f1s


def train_one(pretrain, train_set, val_set, test_set, device):
    print(f"\n{'=' * 60}\nTraining classifier | encoder pretrain: {pretrain}\n{'=' * 60}")
    set_seed(SEED)
    model = build_classifier(pretrain).to(device)

    class_counts = Counter(s[1] for s in train_set.samples)
    weights = torch.tensor(
        [1.0 / class_counts[0], 1.0 / class_counts[1]], device=device
    )
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LR},
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)

    best_f1 = -1.0
    best_path = OUT_DIR / f"classifier_{pretrain}.pth"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[{pretrain}] {epoch}/{EPOCHS}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * imgs.size(0)

        val_acc, val_f1, _ = evaluate(model, val_loader, device)
        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            marker = "  <- saved"
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss {running / len(train_set):.4f} | val_acc {val_acc:.4f} | val_macroF1 {val_f1:.4f}{marker}")

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_acc, test_f1, f1s = evaluate(model, test_loader, device)
    print(f"[{pretrain}] TEST | acc {test_acc:.4f} | macroF1 {test_f1:.4f} | F1 ship {f1s[0]:.4f} | F1 plane {f1s[1]:.4f}")
    return test_acc, test_f1, f1s


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(exist_ok=True)

    samples = collect_samples()
    counts = Counter(s[1] for s in samples)
    print(f"Dataset: {len(samples)} images | ships {counts[0]} | planes {counts[1]}")

    train_s, val_s, test_s = stratified_split(samples)
    print(f"Split: train {len(train_s)} | val {len(val_s)} | test {len(test_s)}")
    print(f"Test class counts: {Counter(s[1] for s in test_s)}")

    train_set = SonarObjectsDataset(train_s, train_mode=True)
    val_set = SonarObjectsDataset(val_s, train_mode=False)
    test_set = SonarObjectsDataset(test_s, train_mode=False)

    results = {}
    for pretrain in ["wreck", "imagenet"]:
        results[pretrain] = train_one(pretrain, train_set, val_set, test_set, device)

    print(f"\n{'=' * 60}\nSUMMARY (test split)\n{'=' * 60}")
    for pretrain, (acc, f1, f1s) in results.items():
        print(f"{pretrain:10s} | acc {acc:.4f} | macroF1 {f1:.4f} | shipF1 {f1s[0]:.4f} | planeF1 {f1s[1]:.4f}")


if __name__ == "__main__":
    main()
