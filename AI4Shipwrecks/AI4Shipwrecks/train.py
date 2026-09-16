import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
OUT_DIR = ROOT / "runs"

IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 60
LR = 2e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.15
SEED = 42
NUM_WORKERS = 0
THRESHOLD = 0.5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pair(image_path, mask_path):
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    mask = (np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.float32)
    return (
        torch.from_numpy(img)[None, :, :],
        torch.from_numpy(mask),
    )


class ShipwreckDataset(Dataset):
    def __init__(self, root):
        self.image_paths = sorted((Path(root) / "images").glob("*.png"))
        self.mask_paths = sorted((Path(root) / "labels").glob("*.png"))
        assert len(self.image_paths) == len(self.mask_paths) and len(self.image_paths) > 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return load_pair(self.image_paths[idx], self.mask_paths[idx])


def site_key(path):
    name = Path(path).stem
    return name.rsplit("_", 1)[0]


def make_split(dataset):
    sites = sorted({site_key(p) for p in dataset.image_paths})
    rng = random.Random(SEED)
    rng.shuffle(sites)
    n_val = max(2, int(len(sites) * VAL_FRACTION))
    val_sites = set(sites[:n_val])
    train_idx = [i for i, p in enumerate(dataset.image_paths) if site_key(p) not in val_sites]
    val_idx = [i for i, p in enumerate(dataset.image_paths) if site_key(p) in val_sites]
    return train_idx, val_idx


class SubsetDataset(Dataset):
    def __init__(self, base, indices, train_mode):
        self.base = base
        self.indices = indices
        self.train_mode = train_mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.base[self.indices[idx]]
        _, h, w = img.shape
        if self.train_mode:
            scale = random.uniform(0.35, 1.0)
            ch = min(h, max(32, int(h * scale)))
            cw = min(w, max(32, int(w * scale)))
            y0 = random.randint(0, h - ch)
            x0 = random.randint(0, w - cw)
            img = img[:, y0 : y0 + ch, x0 : x0 + cw]
            mask = mask[y0 : y0 + ch, x0 : x0 + cw]

        img = F.interpolate(
            img[None],
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        mask = F.interpolate(mask[None, None], size=(IMG_SIZE, IMG_SIZE), mode="nearest")[0]

        if self.train_mode:
            if random.random() < 0.25:
                img = img[None]
                k = 2 * random.randint(1, 2) - 1
                img = F.avg_pool2d(img, k, stride=1, padding=k // 2)
                img = img[0]
            if random.random() < 0.5:
                img = torch.flip(img, [-1])
                mask = torch.flip(mask, [-1])
            if random.random() < 0.5:
                img = torch.flip(img, [-2])
                mask = torch.flip(mask, [-2])
            k = random.randint(0, 3)
            img = torch.rot90(img, k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])
            img = img * random.uniform(0.75, 1.25) + random.uniform(-0.15, 0.15) + torch.randn_like(img) * 0.02

        return img.clamp_(0.0, 1.0), mask


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_inter = 0.0
    total_union = 0.0
    loss_sum = 0.0
    n = 0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        preds = (logits.sigmoid() > THRESHOLD).float()

        inter = (preds * masks).sum().item()
        union = ((preds + masks) > 0).float().sum().item()
        total_inter += inter
        total_union += union
        loss_sum += loss.item() * imgs.size(0)
        n += imgs.size(0)

    iou = total_inter / max(total_union, 1e-7)
    dice = 2 * total_inter / max(total_inter + (total_union - total_inter) + total_inter, 1e-7)
    return loss_sum / n, iou, dice


def criterion(logits, targets):
    return smp.losses.DiceLoss(mode="binary", from_logits=True)(
        logits, targets
    ) + F.binary_cross_entropy_with_logits(logits, targets)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(exist_ok=True)

    base_train = ShipwreckDataset(TRAIN_DIR)
    base_test = ShipwreckDataset(TEST_DIR)

    train_idx, val_idx = make_split(base_train)
    train_set = SubsetDataset(base_train, train_idx, train_mode=True)
    val_set = SubsetDataset(base_train, val_idx, train_mode=False)
    test_set = SubsetDataset(base_test, list(range(len(base_test))), train_mode=False)

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)} | Test samples: {len(test_set)}")

    common = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, **common)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        decoder_attention_type="scse",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_iou = -1.0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / len(train_set)
        val_loss, val_iou, val_dice = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        marker = ""
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder": "resnet34",
                    "img_size": IMG_SIZE,
                    "epoch": epoch,
                    "val_iou": val_iou,
                },
                OUT_DIR / "best_model.pth",
            )
            marker = "  <- saved"

        print(
            f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
            f"| IoU {val_iou:.4f} | Dice {val_dice:.4f}{marker}"
        )

    ckpt = torch.load(OUT_DIR / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_iou, test_dice = evaluate(model, test_loader, device)
    print(f"\nBest epoch: {ckpt['epoch']} (val IoU {ckpt['val_iou']:.4f})")
    print(f"TEST | loss {test_loss:.4f} | IoU {test_iou:.4f} | Dice {test_dice:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["val_iou"])
    axes[1].set_title("Val IoU")
    axes[2].plot(history["val_dice"])
    axes[2].set_title("Val Dice")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "curves.png", dpi=150)


if __name__ == "__main__":
    main()
