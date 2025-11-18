import os
import glob
import re
import json

import cv2
import numpy as np

# ===========================
# 1) 스케일 (필요하면 숫자만 바꾸면 됨)
# ===========================
SLICE_SPACING_M = 0.05   # 슬라이스 간 간격 (X축, 5cm 가정)
SCAN_LENGTH_M   = 10.0   # 한 이미지의 진행 방향 길이 (Y축)
MAX_DEPTH_M     = 5.0    # 최대 깊이 (Z축)

# ===========================
# 2) 경로 설정
# ===========================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_DIR    = os.path.join(BASE_DIR, "continuous_data")  # 연속 YZ 이미지 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] BASE_DIR:", BASE_DIR)
print("[INFO] IMG_DIR :", IMG_DIR)
print("[INFO] OUTPUT  :", OUTPUT_DIR)

# ===========================
# 3) 파일 정렬 (마지막 숫자 기준)
# ===========================
def extract_index(path):
    """
    cavity_yz_MALA_000123.jpg -> 123
    """
    name = os.path.basename(path)
    m = re.search(r"(\d+)(?=\D*$)", name)
    return int(m.group(1)) if m else -1

img_paths = sorted(
    glob.glob(os.path.join(IMG_DIR, "*.jpg")),
    key=extract_index
)

if not img_paths:
    raise RuntimeError(f"이미지를 찾을 수 없습니다: {IMG_DIR}")

print(f"[INFO] 슬라이스 개수: {len(img_paths)}")
print("[INFO] 첫 5개 예시:")
for p in img_paths[:5]:
    print("  -", os.path.basename(p))

# ===========================
# 4) 3D 볼륨 생성
# ===========================
# 첫 이미지로 크기 확인
first = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
if first is None:
    raise RuntimeError(f"이미지를 읽을 수 없습니다: {img_paths[0]}")

H, W = first.shape[:2]
print(f"[INFO] 이미지 크기: H={H}, W={W}")

volume = np.zeros((len(img_paths), H, W), dtype=np.float32)

for i, path in enumerate(img_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] 읽기 실패, 스킵: {path}")
        continue

    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    volume[i] = img

# ===========================
# 5) intensity 정규화
# ===========================
v_min, v_max = volume.min(), volume.max()
volume_norm = (volume - v_min) / (v_max - v_min + 1e-8)

print(f"[INFO] intensity range: min={v_min}, max={v_max}")
print("[INFO] volume_norm shape:", volume_norm.shape)  # (슬라이스, 깊이, 진행)

# ===========================
# 6) 파일로 저장 (시뮬레이션용)
# ===========================
vol_path  = os.path.join(OUTPUT_DIR, "gpr_volume_norm.npy")
meta_path = os.path.join(OUTPUT_DIR, "gpr_volume_meta.json")

np.save(vol_path, volume_norm)

meta = {
    "shape": {
        "slices": int(volume_norm.shape[0]),
        "depth_px": int(volume_norm.shape[1]),
        "length_px": int(volume_norm.shape[2]),
    },
    "spacing": {
        "dx_m": SLICE_SPACING_M,   # 슬라이스 간격 (X)
        "dy_m": SCAN_LENGTH_M,     # 전체 길이 (Y) → 0~SCAN_LENGTH_M 로 스케일
        "dz_m": MAX_DEPTH_M        # 최대 깊이 (Z) → 0~MAX_DEPTH_M 로 스케일
    },
    "raw_intensity_min": float(v_min),
    "raw_intensity_max": float(v_max)
}

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("[INFO] 3D 볼륨 저장 완료:")
print("  -", vol_path)
print("  -", meta_path)

# ===========================
# 7) 간단 point cloud도 하나 만들어두기 (옵션)
# ===========================
THRESH = 0.7  # 밝은 부분만 추출 (필요하면 조정)

mask = volume_norm > THRESH
idx  = np.argwhere(mask)  # (N, 3) = (slice, z, y)

if idx.size == 0:
    print("[WARN] threshold가 너무 높아서 voxel이 없습니다. THRESH를 낮춰보세요.")
else:
    S, H, W = volume_norm.shape

    s_idx = idx[:, 0]
    z_idx = idx[:, 1]
    y_idx = idx[:, 2]

    # index -> 실제 좌표 (x, y, z)
    x = s_idx * SLICE_SPACING_M
    y = (y_idx / max(W - 1, 1)) * SCAN_LENGTH_M
    z = (z_idx / max(H - 1, 1)) * MAX_DEPTH_M

    points = np.stack([x, y, z], axis=1)
    pc_path = os.path.join(OUTPUT_DIR, "gpr_points_thresh070.xyz")

    # x y z intensity 형식으로 저장
    intens = volume_norm[mask].reshape(-1, 1)
    xyz_i  = np.concatenate([points, intens], axis=1)
    np.savetxt(pc_path, xyz_i, fmt="%.5f")

    print(f"[INFO] point cloud 저장 완료 (x y z intensity): {pc_path}")
    print(f"[INFO] 총 포인트 수: {xyz_i.shape[0]}")
