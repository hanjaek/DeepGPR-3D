import numpy as np
from pathlib import Path
import pyvista as pv

THIS_DIR = Path(__file__).resolve().parent
VOLUME_PATH = THIS_DIR / "cavity_volume.npy"

# ------------------------------------------------
# 실제/상대 스케일 설정 (필요 시 값만 바꿔서 재사용)
# ------------------------------------------------
# Z: 깊이 방향, Y: 폭, X: 진행 방향(도로 길이)
PIXEL_SPACING_Z = 1.0   # 이미지 세로 방향(z) 간격
PIXEL_SPACING_Y = 1.0   # 이미지 가로 방향(y) 간격
SLICE_SPACING_X = 0.5   # 슬라이스 간격(x) - 실제 GPR 간격에 맞게


def main():
    # 1) 3D 볼륨 로드 (z, y, x)
    vol_zyx = np.load(VOLUME_PATH).astype(np.float32)
    nz, ny, nx = vol_zyx.shape
    print(f"[INFO] Loaded volume (z, y, x): {vol_zyx.shape}")

    # 2) PyVista는 (x, y, z) 순서를 기대 → 축 재배열
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))  # (x, y, z)

    # 3) ImageData(UniformGrid 역할) 생성
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (SLICE_SPACING_X, PIXEL_SPACING_Y, PIXEL_SPACING_Z)
    grid.origin = (0.0, 0.0, 0.0)

    # 3D 스칼라 필드로 cavity 값 할당
    grid["cavity"] = vol_xyz.ravel(order="F")

    # 4) 공동 iso-surface 추출
    iso = grid.contour(isosurfaces=[0.5], scalars="cavity")

    # 5) 도로 외곽 박스(겉 껍데기)
    bounds = grid.bounds
    road_box = pv.Box(bounds=bounds)

    # 6) 시각화
    p = pv.Plotter()

    # 도로 껍데기: 회색 반투명
    p.add_mesh(
        road_box,
        color="gray",
        opacity=0.35,
        smooth_shading=True,
    )

    # 내부 공동: 진한 색 메쉬
    p.add_mesh(
        iso,
        color="black",   # 필요하면 "green", "red" 등으로 변경
        opacity=1.0,
        smooth_shading=True,
    )

    p.show_axes()
    p.add_bounding_box()
    p.camera_position = "iso"   # 대각선 위에서 보는 뷰
    p.show()


if __name__ == "__main__":
    main()