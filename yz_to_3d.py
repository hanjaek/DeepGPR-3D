import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# === 1. 스케일 설정 (네가 바꿔도 됨) ===
SLICE_SPACING_M = 0.5   # YZ 슬라이스 사이 간격 (X축)
SCAN_LENGTH_M  = 50.0   # 한 이미지의 진행 방향 길이 (Y축 전체)
MAX_DEPTH_M    = 5.0    # 한 이미지의 깊이 범위 (Z축 전체)

# === 2. 경로 설정 ===
IMG_DIR   = r"C:\Users\hjk25\gpr_to_cavity\test_data"
LABEL_DIR = r"C:\Users\hjk25\gpr_to_cavity\ai_hub\src\yolov5_master\runs\detect\exp4\labels"

image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))  # 필요하면 png도 추가

points = []      # (x,y,z) 포인트 저장
point_labels = []  # 클래스(0=cavity, 1=box 등)

for slice_idx, img_path in enumerate(image_paths):
    img_name = os.path.basename(img_path)
    stem, _ = os.path.splitext(img_name)
    label_path = os.path.join(LABEL_DIR, stem + ".txt")
    if not os.path.exists(label_path):
        continue  # 탐지 결과 없는 이미지

    # 이미지 크기
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape[:2]

    # 이 슬라이스의 X좌표 (왼쪽에서부터 0, 0.5, 1.0, … 이런 식)
    x_coord = slice_idx * SLICE_SPACING_M

    with open(label_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            # --save-conf 썼으니까: class cx cy w h conf
            if len(parts) == 6:
                cls, cx, cy, w, h, conf = parts
            else:  # 혹시 conf 없이 저장됐다면
                cls, cx, cy, w, h = parts
                conf = 1.0

            cls = int(cls)
            cx  = float(cx); cy = float(cy)
            w   = float(w);  h  = float(h)

            # 정규화 → 픽셀
            cx_px = cx * W
            cy_px = cy * H
            w_px  = w * W
            h_px  = h * H

            # B-box 안에서 몇 개 포인트만 샘플링 (격자 3x3 정도)
            nx, ny = 3, 3
            xs = np.linspace(cx_px - w_px/2, cx_px + w_px/2, nx)
            ys = np.linspace(cy_px - h_px/2, cy_px + h_px/2, ny)

            for px in xs:
                for py in ys:
                    # 픽셀 → 실제 좌표
                    y_coord = (px / W) * SCAN_LENGTH_M
                    z_coord = (py / H) * MAX_DEPTH_M

                    points.append([x_coord, y_coord, z_coord])
                    point_labels.append(cls)

points = np.array(points)
point_labels = np.array(point_labels)

print("총 포인트 수:", len(points))

# === 3. 3D 시각화 (간단 버전) ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# cavity/box 색 구분 (0=cavity, 1=box 라고 가정)
colors = np.where(point_labels == 0, 'r', 'b')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c=colors, alpha=0.7)

ax.set_xlabel("X (m)  - 라인 사이 간격")
ax.set_ylabel("Y (m)  - 진행 방향")
ax.set_zlabel("Z (m)  - 깊이")
ax.set_title("GPR YOLO 탐지 결과 3D 포인트 뷰")

plt.tight_layout()
plt.show()
