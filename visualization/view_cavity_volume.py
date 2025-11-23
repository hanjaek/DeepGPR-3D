# visualization/view_cavity_volume.py

import numpy as np
from pathlib import Path
import pyvista as pv

THIS_DIR = Path(__file__).resolve().parent
VOLUME_PATH = THIS_DIR / "cavity_volume_test1.npy"

# -----------------------------
# 실제 스케일 설정 (단위: m)
# -----------------------------
# z: 깊이 방향 (한 픽셀당 몇 m인지 모르면 일단 0.05 같은 값으로 가정하고
# 나중에 GPR 스펙 보고 수정)
PIXEL_SPACING_Z = 0.05  # 예: 5cm (필요시 수정)
PIXEL_SPACING_Y = 0.05  # 예: 5cm (필요시 수정)

# x: 슬라이스 사이 간격 = 50cm
SLICE_SPACING_X = 0.5   # 50cm


def main():
    # 1) 볼륨 로드 (z, y, x)
    vol = np.load(VOLUME_PATH).astype(np.float32)
    nz, ny, nx = vol.shape
    print(f"[INFO] Loaded volume (z, y, x): {vol.shape}")

    # 2) PyVista는 (x, y, z) 순서를 기대하므로 transpose
    vol_xyz = np.transpose(vol, (2, 1, 0))  # (x, y, z)

    # 3) UniformGrid 생성
    grid = pv.UniformGrid()
    grid.dimensions = (nx, ny, nz)  # (nx, ny, nz)
    grid.spacing = (SLICE_SPACING_X, PIXEL_SPACING_Y, PIXEL_SPACING_Z)
    grid.origin = (0.0, 0.0, 0.0)

    # 0/1 마스크인데, 나중에 probability로 바꿀 수도 있으니 이름은 cavity 로 둠
    grid["cavity"] = vol_xyz.ravel(order="F")

    # 4) iso-surface 추출 (공동 경계만 메쉬로)
    # 0.5 기준 → 0/1 데이터에서 0.5 이상은 공동
    iso = grid.contour(isosurfaces=[0.5], scalars="cavity")

    # 5) 시각화
    p = pv.Plotter()
    # 반투명 볼륨 (지반 느낌)
    p.add_volume(
        grid,
        scalars="cavity",
        opacity=[0, 0, 0.1, 0.3, 0.6, 1.0],  # 간단 커스텀 opacity transfer
        clim=[0, 1],
        shade=True,
    )
    # 공동 iso-surface
    p.add_mesh(
        iso,
        color="green",
        opacity=1.0,
    )

    p.show_axes()
    p.add_bounding_box()
    p.show()


if __name__ == "__main__":
    main()
