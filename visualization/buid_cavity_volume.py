import numpy as np
from pathlib import Path
from PIL import Image

# ================================================================
# 1) 경로 설정
#    - 본 스크립트가 있는 위치 기준으로 프로젝트 루트를 찾고,
#      그 아래 test/test1 폴더의 마스크 이미지들을 자동으로 읽는다.
# ================================================================
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
TEST_DIR = PROJECT_ROOT / "test" / "test1"

# 지원하는 이미지 확장자
IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]

# 출력 파일 (3D 볼륨)
OUT_PATH = THIS_DIR / "cavity_volume_test1.npy"


def load_mask_slices(test_dir: Path) -> np.ndarray:
    """
    마스크 이미지(YZ 슬라이스)들을 읽어 X 방향으로 쌓아
    하나의 3D 볼륨 (z, y, x) 을 생성한다.

    - 하나의 PNG/JPG 이미지는 YZ 평면이라고 가정한다.
      (세로: Z축 / 가로: Y축)
    - 파일명 순서대로 읽어 X축을 구성한다.
    """
    # test1 안의 이미지들 중 확장자 필터링 후 오름차순 정렬
    img_paths = sorted(
        [p for p in test_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    )

    if not img_paths:
        raise FileNotFoundError(f"No image files found in {test_dir}")

    masks = []

    for p in img_paths:
        img = Image.open(p).convert("L")   # Grayscale (0~255)
        arr = np.array(img, dtype=np.float32)

        # 흰색(>0)을 공동(1), 검정(0)을 지반(0)으로 처리
        mask = (arr > 0).astype(np.uint8)  # shape: (z, y)

        masks.append(mask)

    # 이미지들을 X축 방향으로 stack → 3D 볼륨 생성
    vol = np.stack(masks, axis=-1)  # (z, y, x)

    print(f"[INFO] Loaded {len(img_paths)} slices")
    print(f"[INFO] Volume shape (z, y, x): {vol.shape}")
    return vol


def main():
    # 2D → 3D 변환
    vol = load_mask_slices(TEST_DIR)

    # Numpy 파일로 3D 볼륨 저장
    np.save(OUT_PATH, vol)
    print(f"[INFO] Saved volume to: {OUT_PATH}")


if __name__ == "__main__":
    main()
