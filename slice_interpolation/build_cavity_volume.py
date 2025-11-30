import numpy as np
from pathlib import Path
import cv2

from sdt_interpolation import build_volume_from_masks

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
TEST_DIR = ROOT_DIR / "test_interpolation"

OUT_RAW = THIS_DIR / "cavity_volume_raw.npy"
OUT_INTERP = THIS_DIR / "cavity_volume_interp.npy"


def load_masks_all():
    # test_interpolation 안의 PNG 전부 사용 (10장)
    img_paths = sorted(TEST_DIR.glob("*.png"))
    if len(img_paths) < 2:
        raise RuntimeError("test_interpolation 폴더에 최소 2장 이상 필요합니다.")

    masks = []
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없음: {p}")
        _, mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        masks.append(mask)
        print(f"[INFO] Loaded {p.name}, shape={mask.shape}, unique={np.unique(mask)}")

    return masks


def main():
    masks_zy = load_masks_all()
    nz, ny = masks_zy[0].shape
    n_slices = len(masks_zy)
    print(f"[INFO] Loaded {n_slices} masks")

    # 1) 원본 N장 그대로 쌓은 볼륨 (비교용)
    vol_raw = np.zeros((nz, ny, n_slices), dtype=np.uint8)
    for i, m in enumerate(masks_zy):
        vol_raw[:, :, i] = m
    np.save(OUT_RAW, vol_raw)
    print(f"[INFO] Saved RAW volume: {OUT_RAW}, shape={vol_raw.shape}")

    # 2) SDT 보간된 볼륨
    vol_interp, num_mid = build_volume_from_masks(
        masks_zy,
        orig_spacing_x=0.5,    # 슬라이스 간 실제 간격 50cm
        target_spacing_x=0.02  # 보간 간격 (예: 5cm, 필요하면 0.1이나 0.02로 조절)
    )
    np.save(OUT_INTERP, vol_interp)
    print(f"[INFO] Saved INTERP volume: {OUT_INTERP}, shape={vol_interp.shape}")
    print(f"[INFO] num_mid between slices: {num_mid}")


if __name__ == "__main__":
    main()
