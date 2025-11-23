# visualization/build_cavity_volume.py

import numpy as np
from pathlib import Path
from PIL import Image

# -----------------------------
# 설정
# -----------------------------
# 이 스크립트가 어디 있든 상관없이, 프로젝트 루트 기준으로 test/test1을 찾도록 함
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
TEST_DIR = PROJECT_ROOT / "test" / "test1"

# 마스크 이미지 확장자들 (필요하면 추가)
IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]

# 출력 npy 경로
OUT_PATH = THIS_DIR / "cavity_volume_test1.npy"

# 슬라이스 간격 (x 방향, 단위: m)
SLICE_SPACING_X = 0.5  # 50cm


def load_mask_slices(test_dir: Path) -> np.ndarray:
    """
    test/test1 안의 마스크 이미지들을 읽어서
    (z, y, x) 형태의 3D 볼륨으로 합친다.
    - z: 이미지의 세로 방향 (깊이)
    - y: 이미지의 가로 방향 (탐사 라인에 수직)
    - x: 슬라이스 인덱스 (탐사 진행 방향)
    """
    # test1 안의 이미지 파일들 정렬해서 읽기
    img_paths = sorted(
        [p for p in test_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    )

    if not img_paths:
        raise FileNotFoundError(f"No image files found in {test_dir}")

    masks = []

    for p in img_paths:
        img = Image.open(p).convert("L")  # Grayscale
        arr = np.array(img, dtype=np.float32)

        # 0/1 마스크로 변환 (필요에 따라 threshold 조절 가능)
        # 0이 배경(지반), >0 이 공동이라고 가정
        mask = (arr > 0).astype(np.uint8)  # (H, W) = (z, y)

        masks.append(mask)

    # masks: list of (z, y) -> stack along x-axis
    vol = np.stack(masks, axis=-1)  # (z, y, x)

    print(f"[INFO] Loaded {len(img_paths)} slices from {test_dir}")
    print(f"[INFO] Volume shape (z, y, x): {vol.shape}")
    return vol


def main():
    vol = load_mask_slices(TEST_DIR)

    # 필요하면 여기서 추후 "probability"나 "distance transform" 등으로 변환 가능
    # 지금은 0/1 이진 마스크 그대로 저장
    np.save(OUT_PATH, vol)
    print(f"[INFO] Saved volume to: {OUT_PATH}")


if __name__ == "__main__":
    main()
