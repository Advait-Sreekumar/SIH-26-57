from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import train

OUT_DIR = Path(__file__).resolve().parent / "runs"
THRESHOLD = 0.5


def has_scse(state_dict):
    return any("attention" in k for k in state_dict.keys())


def build_model(encoder, scse):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type="scse" if scse else None,
    )


@torch.no_grad()
def run_eval(ckpt_path, loader, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(ckpt.get("encoder", "resnet34"), has_scse(ckpt["model_state_dict"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    inter = union = 0.0
    site_stats = {}
    preds_store = []
    for imgs, masks, names in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = (logits.sigmoid() > THRESHOLD).float().cpu()
        masks = masks.cpu()

        for i in range(imgs.size(0)):
            p, m = preds[i], masks[i]
            pi = (p * m).sum().item()
            pu = ((p + m) > 0).float().sum().item()
            inter += pi
            union += pu
            site = train.site_key(names[i])
            s = site_stats.setdefault(site, [0.0, 0.0])
            s[0] += pi
            s[1] += pu
            preds_store.append((names[i], imgs[i].cpu(), m, p))

    iou = inter / max(union, 1e-7)
    dice = 2 * inter / max(inter + union, 1e-7)
    site_iou = {k: v[0] / max(v[1], 1e-7) for k, v in sorted(site_stats.items())}
    return model, iou, dice, site_iou, preds_store


class EvalDataset(train.SubsetDataset):
    def __getitem__(self, idx):
        img, mask = self.base[self.indices[idx]]
        img = F.interpolate(
            img[None], size=(train.IMG_SIZE, train.IMG_SIZE),
            mode="bilinear", align_corners=False, antialias=True,
        )[0]
        mask = F.interpolate(mask[None, None], size=(train.IMG_SIZE, train.IMG_SIZE), mode="nearest")[0]
        name = Path(self.base.image_paths[self.indices[idx]]).name
        return img, mask, name


def visualize(store, path, n=8):
    picks = [store[i] for i in np.linspace(0, len(store) - 1, n).astype(int)]
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    for r, (name, img, gt, pred) in enumerate(picks):
        axes[r][0].imshow(img[0], cmap="gray")
        axes[r][0].set_ylabel(name[:28], fontsize=7)
        axes[r][1].imshow(gt[0], cmap="gray")
        axes[r][2].imshow(pred[0], cmap="gray")
        for c, t in zip(range(3), ["Sonar", "Ground truth", "Prediction"]):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            axes[r][c].set_title(t if r == 0 else "")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_test = train.ShipwreckDataset(train.TEST_DIR)
    test_set = EvalDataset(base_test, list(range(len(base_test))), train_mode=False)
    loader = DataLoader(test_set, batch_size=train.BATCH_SIZE, num_workers=0)

    print("=" * 60)
    _, iou, dice, site_iou, store = run_eval(OUT_DIR / "best_model.pth", loader, device)
    print(f"\nRun 2 (best_model.pth)")
    print(f"TEST IoU : {iou:.4f}")
    print(f"TEST Dice: {dice:.4f}")

    v1_path = OUT_DIR / "best_model_v1_iou0.71.pth"
    if v1_path.exists():
        _, iou1, dice1, site_iou1, _ = run_eval(v1_path, loader, device)
        print(f"\nRun 1 (baseline)")
        print(f"TEST IoU : {iou1:.4f}")
        print(f"TEST Dice: {dice1:.4f}")

        print("\nPer-site IoU (run2 | run1):")
        for site in site_iou:
            print(f"  {site:24s} {site_iou[site]:.4f} | {site_iou1[site]:.4f}")

    visualize(store, OUT_DIR / "test_results.png")
    print(f"\nSaved visualizations -> {OUT_DIR / 'test_results.png'}")


if __name__ == "__main__":
    main()
