import os
import json
import numpy as np
import pyvista as pv
from pyvista import UniformGrid   # ★ 핵심: 여기서 직접 import

# -----------------------
# paths
# -----------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# use preprocessed volume if exists
vol_path  = os.path.join(OUTPUT_DIR, "gpr_volume_preprocessed.npy")
meta_path = os.path.join(OUTPUT_DIR, "gpr_volume_preprocessed_meta.json")

if not os.path.exists(vol_path):
    print("[WARN] preprocessed volume not found, using raw volume.")
    vol_path  = os.path.join(OUTPUT_DIR, "gpr_volume_norm.npy")
    meta_path = os.path.join(OUTPUT_DIR, "gpr_volume_meta.json")

volume = np.load(vol_path)  # shape: (S, H, W)
with open(meta_path, "r") as f:
    meta = json.load(f)

print("Loaded volume:", volume.shape)

S, H, W = volume.shape

dx = meta["spacing"].get("dx_m", 1.0)
dy = meta["spacing"].get("dy_m", 1.0)
dz = meta["spacing"].get("dz_m", 1.0)

# -----------------------
# build UniformGrid
# -----------------------
# 여기서는 X = slices, Y = along-track, Z = depth로 매핑
nx, ny, nz = S, W, volume.shape[1]

grid = UniformGrid()              # ★ pv.UniformGrid() 대신
grid.dimensions = (nx, ny, nz)    # number of points in x, y, z
grid.spacing    = (dx, dy, dz)    # 실제 물리 간격
grid.origin     = (0.0, 0.0, 0.0)

# PyVista는 (nx, ny, nz) 순서의 C-order flat array를 기대하므로
# 현재 volume: (S, H, W) = (X, Z, Y)
# → (X, Y, Z)로 transpose: (S, W, H)
scalars = volume.transpose(0, 2, 1).ravel(order="C")
grid["Intensity"] = scalars

# -----------------------
# volume rendering
# -----------------------
pl = pv.Plotter()
pl.add_volume(
    grid,
    scalars="Intensity",
    opacity="sigmoid",   # 투명도 곡선
    cmap="viridis",      # 색맵 (원하면 inferno, plasma 등으로 바꿔도 됨)
)

pl.add_axes(line_width=2)
pl.add_bounding_box()
pl.add_text("3D GPR Volume", font_size=12)

pl.show()
