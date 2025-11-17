import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------
# paths
# -----------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

vol_path  = os.path.join(OUTPUT_DIR, "gpr_volume_norm.npy")
meta_path = os.path.join(OUTPUT_DIR, "gpr_volume_meta.json")

# load 3D volume + metadata
volume = np.load(vol_path)
with open(meta_path, "r") as f:
    meta = json.load(f)

print("Loaded volume:", volume.shape)

# -----------------------
# threshold for point cloud
# -----------------------
THRESH = 0.7
mask = volume > THRESH
idx  = np.argwhere(mask)

if idx.size == 0:
    print("No voxels above threshold. Try lower THRESH.")
    exit()

s_idx = idx[:, 0]
z_idx = idx[:, 1]
y_idx = idx[:, 2]

dx = meta["spacing"]["dx_m"]
dy = meta["spacing"]["dy_m"]
dz = meta["spacing"]["dz_m"]

# index → actual coordinates
x = s_idx * dx
y = y_idx * dy
z = z_idx * dz

intens = volume[mask].ravel()

# -----------------------
# 3D visualization
# -----------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

max_points = 50000
if len(x) > max_points:
    idx_sample = np.random.choice(len(x), max_points, replace=False)
    x = x[idx_sample]
    y = y[idx_sample]
    z = z[idx_sample]
    intens = intens[idx_sample]

scatter = ax.scatter(x, y, z, c=intens, s=1, cmap="turbo")

ax.set_xlabel("X (slice direction)")
ax.set_ylabel("Y (along-track)")
ax.set_zlabel("Z (depth)")
ax.set_title("3D GPR Volume (Thresholded Points)")

plt.colorbar(scatter, shrink=0.5, label="Intensity")
plt.show()