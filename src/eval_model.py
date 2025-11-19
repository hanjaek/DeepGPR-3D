# eval_unet.py
import torch
from torch.utils.data import DataLoader
from dataset import GPRCavityDataset
from model import UNet
from train import compute_metrics  # 아까 정의한 함수 재사용

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_root = "../data2"
    mask_root = "../data2_mask"
    dataset = GPRCavityDataset(img_root, mask_root, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load("../checkpoints/unet_best.pth", map_location=device))
    model.eval()

    dice_sum = iou_sum = acc_sum = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            dice, iou, acc = compute_metrics(logits, masks, threshold=0.5)
            dice_sum += dice
            iou_sum  += iou
            acc_sum  += acc

    n = len(loader)
    print(f"Dice={dice_sum/n:.4f}, IoU={iou_sum/n:.4f}, PixelAcc={acc_sum/n:.4f}")

if __name__ == "__main__":
    main()
