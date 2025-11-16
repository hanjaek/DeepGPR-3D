import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1) 스케일 & 경로 설정
# ===========================
SLICE_SPACING_M = 0.5   # 슬라이스 간 간격 (X축)
SCAN_LENGTH_M   = 10.0  # 진행 방향 길이 (Y축)
MAX_DEPTH_M     = 5.0   # 최대 깊이 (Z축)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_DIR    = os.path.join(BASE_DIR, "test_data")
LABEL_DIR  = os.path.join(BASE_DIR, "ai_hub/src/yolov5_master/runs/detect/exp2/labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# 2) 017_* 이미지 목록
# ===========================
img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "017_*.jpg")))
if not img_paths:
    raise RuntimeError(f"017_* 이미지를 찾을 수 없습니다: {IMG_DIR}")

print("[INFO] 사용할 슬라이스:")
for p in img_paths:
    print("  -", os.path.basename(p))

num_slices = len(img_paths)

# YOLO detect가 기본 640x576으로 리사이즈해서 처리했으니 이렇게 가정
H = 576  # 세로(깊이 방향 픽셀)
W = 640  # 가로(진행 방향 픽셀)

# ===========================
# 3) cavity 볼륨 생성
# ===========================
volume_mask = np.zeros((num_slices, H, W), dtype=bool)

for slice_idx, img_path in enumerate(img_paths):
    img_name = os.path.basename(img_path)
    stem, _ = os.path.splitext(img_name)
    label_path = os.path.join(LABEL_DIR, stem + ".txt")

    if not os.path.exists(label_path):
        print(f"[INFO] 라벨 없음, 스킵: {label_path}")
        continue

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # --save-conf 사용: class cx cy w h conf
            if len(parts) == 6:
                cls, cx, cy, w, h, conf = parts
            else:
                cls, cx, cy, w, h = parts
                conf = 1.0

            cls = int(cls)
            cx  = float(cx)
            cy  = float(cy)
            w   = float(w)
            h   = float(h)

            # 👉 일단 데모용: class 상관 없이 전부 cavity처럼 시각화
            # 나중에 0=box, 1=cavity 매핑 확실히 알게 되면 여기서 필터링
            # 예시: CAVITY_CLASSES = [1];  if cls not in CAVITY_CLASSES: continue

            # YOLO 정규화 좌표 → 픽셀 좌표
            cx_px = cx * W
            cy_px = cy * H
            w_px  = w * W
            h_px  = h * H

            x1 = int(max(cx_px - w_px / 2, 0))
            x2 = int(min(cx_px + w_px / 2, W - 1))
            y1 = int(max(cy_px - h_px / 2, 0))
            y2 = int(min(cy_px + h_px / 2, H - 1))

            volume_mask[slice_idx, y1:y2+1, x1:x2+1] = True

voxel_count = volume_mask.sum()
print("[INFO] cavity voxel 개수:", voxel_count)
if voxel_count == 0:
    print("[WARN] cavity 표시된 voxel이 없습니다. 그래도 그림은 시도합니다.")

# ===========================
# 4) voxel 중심들을 3D 점으로 찍어서 시각화
# ===========================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

if voxel_count > 0:
    # 너무 많으면 샘플링 (속도/가독성 조절용)
    step = max(1, H // 60)  # 예: H=576이면 step ≈ 9
    cav_small = volume_mask[:, ::step, ::step]
    S, Hs, Ws = cav_small.shape

    idx = np.argwhere(cav_small)  # (N, 3) : (slice, z_idx, y_idx)
    s_idx = idx[:, 0]
    z_idx = idx[:, 1]
    y_idx = idx[:, 2]

    # 인덱스를 실제 m 단위 좌표로 변환
    x = s_idx * SLICE_SPACING_M
    y = (y_idx / max(Ws - 1, 1)) * SCAN_LENGTH_M
    z = (z_idx / max(Hs - 1, 1)) * MAX_DEPTH_M

    ax.scatter(x, y, z,
               s=5,
               c='red',
               alpha=0.7,
               marker='o')

# 지반 큐브 외곽선 그리기 (시각적 참고용)
max_x = (num_slices - 1) * SLICE_SPACING_M
max_y = SCAN_LENGTH_M
max_z = MAX_DEPTH_M

# 모서리 선들
for x0 in [0, max_x]:
    ax.plot([x0, x0], [0, 0], [0, max_z], color='gray', alpha=0.4)
    ax.plot([x0, x0], [max_y, max_y], [0, max_z], color='gray', alpha=0.4)
for y0 in [0, max_y]:
    ax.plot([0, max_x], [y0, y0], [0, 0], color='gray', alpha=0.4)
    ax.plot([0, max_x], [y0, y0], [max_z, max_z], color='gray', alpha=0.4)
for z0 in [0, max_z]:
    ax.plot([0, 0], [0, max_y], [z0, z0], color='gray', alpha=0.4)
    ax.plot([max_x, max_x], [0, max_y], [z0, z0], color='gray', alpha=0.4)

ax.set_xlabel("X (m) - 슬라이스 방향 (017_1, 017_2, ...)")
ax.set_ylabel("Y (m) - 진행 방향 (~10 m)")
ax.set_zlabel("Z (m) - 깊이 (~5 m)")
ax.set_title("Site 017 기반 cavity 3D 시각화 (voxel 포인트 클라우드)")
ax.invert_zaxis()  # 깊이가 아래로 보이게

out_path = os.path.join(OUTPUT_DIR, "3d_cavity_point_017.png")
plt.savefig(out_path, dpi=300)
print("[INFO] 저장:", out_path)

plt.tight_layout()
plt.show()
