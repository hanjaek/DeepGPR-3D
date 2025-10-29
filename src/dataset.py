import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class GPRCavityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir 구조 예시:
            data/
              images/
                001_1.jpg
                001_2.jpg
              masks/
                001_1_mask.png
                001_2_mask.png
        """
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.transform = transform

        # ---------------- 파일 리스트 로드 ----------------
        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]           # e.g. "001_1.jpg"
        base = os.path.splitext(img_name)[0]     # "001_1"
        mask_name = base + "_mask.png"           # "001_1_mask.png"

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # ---------------- 이미지 & 마스크 로드 ----------------
        image = Image.open(img_path).convert("RGB")  # (H,W,3)
        mask = Image.open(mask_path).convert("L")    # (H,W) 0~255

        image_np = np.array(image, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 0).astype(np.float32)   # 0/1로 변환

        # ---------------- 데이터 변환 (증강 등) ----------------
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented["image"]
            mask_np = augmented["mask"]

        # ---------------- Tensor 변환 ----------------
        image_t = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
        mask_t = torch.tensor(mask_np).unsqueeze(0).float()

        return image_t, mask_t
