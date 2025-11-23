import numpy as np
from pathlib import Path
import pyvista as pv

THIS_DIR = Path(__file__).resolve().parent
VOLUME_PATH = THIS_DIR / "cavity_volume_test1.npy"

# 도로 비율용 스페이싱 (상대 비율만 맞으면 됨)
PIXEL_SPACING_Z = 1.0   # 깊이 방향
PIXEL_SPACING_Y = 1.0   # 폭 방향
SLICE_SPACING_X = 2.0   # 길이 방향(슬라이스 간격, 시각적으로만 길게 보이게 -> 현재는 테스트 이미지가 10장이라 원래는 0.5(50cm))


def main():
    # 1) 볼륨 로드 (z, y, x)
    vol_zyx = np.load(VOLUME_PATH).astype(np.float32)
    nz, ny, nx = vol_zyx.shape
    print(f"[INFO] Loaded volume (z, y, x): {vol_zyx.shape}")

    # 2) PyVista는 (x, y, z) 순서를 기대 → 축 재배열
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))  # (x, y, z)

    # 3) ImageData(= UniformGrid 역할) 생성
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (SLICE_SPACING_X, PIXEL_SPACING_Y, PIXEL_SPACING_Z)
    grid.origin = (0.0, 0.0, 0.0)

    # 3D 스칼라 필드로 cavity 값 할당
    grid["cavity"] = vol_xyz.ravel(order="F")

    # 4) 공동 표면 iso-surface
    iso = grid.contour(isosurfaces=[0.5], scalars="cavity")

    # 5) 도로 외곽 박스
    bounds = grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    road_box = pv.Box(bounds=bounds)

    # 6) 시각화
    p = pv.Plotter()

    # (1) 도로 껍데기: 회색 반투명 박스
    p.add_mesh(
        road_box,
        color="gray",
        opacity=0.35,
        smooth_shading=True,
    )

    # (2) 내부 공동: 진한 색 메쉬
    p.add_mesh(
        iso,
        color="black",   # 필요하면 "green" 등으로 변경
        opacity=1.0,
        smooth_shading=True,
    )

    p.show_axes()
    p.add_bounding_box()
    p.camera_position = "iso"  # 대각선 위에서 보는 뷰
    p.show()


if __name__ == "__main__":
    main()
