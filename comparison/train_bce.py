# ==================================================
# Train UNet for Cavity Segmentation
# - BCE Only
# - No Dice, No Augmentation
# ==================================================

import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from dataset import GPRCavityDataset
from model import UNet

# ============================
# Segmentation Metrics
# ============================
def compute_metrics(logits, masks, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    masks_flat = masks.view(masks.size(0), -1)

    intersection = (preds_flat * masks_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1) - intersection

    eps = 1e-7
    dice = (2 * intersection + eps) / (preds_flat.sum(dim=1) + masks_flat.sum(dim=1) + eps)
    iou = (intersection + eps) / (union + eps)
    acc = (preds_flat == masks_flat).float().mean(dim=1)

    return dice.mean().item(), iou.mean().item(), acc.mean().item()

# ==================================================
# Main Training
# ==================================================
def main():
    print("[INFO] Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] Mode  : BCE Only (No Aug, No Dice)")

    img_root = "../data2"
    mask_root = "../data2_mask"

    batch_size = 4
    lr = 1e-3
    num_epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset without augmentation
    dataset = GPRCavityDataset(
        img_root=img_root,
        mask_root=mask_root,
        transform=None      # ★ no augmentation
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Model & Loss
    model = UNet(n_channels=1, n_classes=1).to(device)
    bce = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = 9999.0

    for epoch in range(1, num_epochs + 1):
        # ---------------- Train ----------------
        model.train()
        train_loss_sum = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = bce(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)

        train_loss = train_loss_sum / n_train

        # ---------------- Validation ----------------
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum  = 0.0
        val_acc_sum  = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)

                loss = bce(logits, masks)
                val_loss_sum += loss.item() * imgs.size(0)

                d, i, a = compute_metrics(logits, masks)
                val_dice_sum += d
                val_iou_sum  += i
                val_acc_sum  += a

        val_loss = val_loss_sum / n_val
        val_dice = val_dice_sum / n_val
        val_iou  = val_iou_sum  / n_val
        val_acc  = val_acc_sum  / n_val

        print(
            f"[Epoch {epoch:03d}] "
            f"Train={train_loss:.4f}  "
            f"Val={val_loss:.4f}  "
            f"Dice={val_dice:.4f}  "
            f"IoU={val_iou:.4f}  "
            f"PixAcc={val_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("../checkpoints", exist_ok=True)
            save_path = "../checkpoints/train_bce.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   -> Best model saved: {save_path}")


if __name__ == "__main__":
    main()
