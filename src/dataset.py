# ==================================================
# GPR Cavity Dataset Loader  (data2 / data2_mask용)
# - 이미지:  data2/cavity_yz_MALA_000228.jpg
# - 마스크: data2_mask/cavity_yz_MALA_000228_mask.jpg
# ==================================================

import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class GPRCavityDataset(Dataset):
    """
    ==================================================
    GPR + Mask 쌍을 로딩하는 Dataset 클래스
    - 이미지:  img_root/*.jpg
    - 마스크:  mask_root/{원본이름}_mask.jpg
    ==================================================
    """

    def __init__(self, img_root, mask_root, transform=None, mask_thresh: int = 10):
        self.samples = []
        self.transform = transform
        self.mask_thresh = mask_thresh

        # img_root 안의 모든 jpg 이미지
        img_paths = sorted(glob.glob(os.path.join(img_root, "*.jpg")))

        for img_path in img_paths:
            fname = os.path.basename(img_path)
            name, ext = os.path.splitext(fname)

            # 마스크 파일 이름: 원본이름 + "_mask" + ext
            mask_fname = f"{name}_mask{ext}"
            mask_path = os.path.join(mask_root, mask_fname)

            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))
            else:
                print(f"[WARN] mask not found for {img_path}")

        print(f"[INFO] Total matched pairs: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # ---------------- 이미지/마스크 로딩 ----------------
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)

        # cavity 영역 이진화
        mask = (mask > self.mask_thresh).astype(np.float32)

        # (H,W) -> (1,H,W)
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        # train.py에서 넘겨준 augmentation 적용
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask