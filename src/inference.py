import os
import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from model import UNet

# ---------------- 이미지 전처리 함수 ----------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    img_t = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
    return img_t.unsqueeze(0)  # [1,3,H,W]

# ---------------- 예측 시각화 ----------------
def predict_and_save(model, image_dir, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ])

    model.eval()
    with torch.no_grad():
        for fname in img_files:
            img_path = os.path.join(image_dir, fname)
            image = load_image(img_path).to(device)

            # 모델 예측
            pred = model(image)
            pred_sig = torch.sigmoid(pred)

            # 임계값 0.5 이상인 부분만 공동(1)
            pred_mask = (pred_sig > 0.5).float()

            # 저장 경로
            save_path = os.path.join(save_dir, fname.replace(".jpg", "_pred.png"))
            save_image(pred_mask, save_path)
            print(f"✅ Saved: {save_path}")

# ---------------- 메인 실행부 ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Using device: {device}")

    # 모델 로드
    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint_path = "./outputs/checkpoints/epoch_50.pth"  # <-- 경로 조정 가능
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"📦 Loaded checkpoint: {checkpoint_path}")

    # 예측 실행
    image_dir = "./data/images"
    save_dir = "./outputs/predictions"
    predict_and_save(model, image_dir, save_dir, device)

    print("🎉 Inference complete! Results saved to outputs/predictions/")

if __name__ == "__main__":
    main()
