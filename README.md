# 🛰️ GPR Cavity Segmentation

> 연속 GPR(B-scan) 데이터에서 **공동(cavity)** 가 포함된 단면을 자동으로 골라내고,  
> 선택된 단면에 대해 **pixel-level cavity mask** 를 생성하는 2-단계 딥러닝 파이프라인

---

## 📖 프로젝트 개요 (Overview)

지표투과레이더(Ground Penetrating Radar, GPR)는 지하 공동(cavity), 공사 구멍, 매설물 등을 탐지할 수 있는 비파괴 검사 기술입니다.  
하지만 수천 장에 이르는 GPR 단면을 **전문가가 한 장씩 눈으로 판독**하는 것은 시간이 많이 들고, 사람마다 결과가 달라질 수 있습니다.

이 프로젝트는 다음과 같은 흐름으로 문제를 해결합니다.

1. **대량 GPR 연속 단면(연속 MALA 데이터)** 에 대해  
   AI Hub GPR 데이터로 학습된 **YOLO 기반 분류·탐지 모델**을 사용해  
   ⇒ 공동(cavity)이 탐지된 단면만 자동으로 골라냄.
2. 골라낸 cavity 단면들을 모아 **세그멘테이션용 데이터셋(data2 / data2_mask)** 을 만들고,
3. **U-Net 기반 segmentation 모델**을 학습하여  
   ⇒ 각 단면에서 cavity의 **정확한 형태를 pixel 단위 mask로 예측**.
4. 예측된 mask는 이후 3D GPR 볼륨/지반 붕괴 시뮬레이션의 입력으로 사용할 수 있도록 설계.

---

## 🔁 전체 파이프라인 (High-Level Pipeline)

1. **연속 GPR 데이터 수집**
   - 예: `continuous_data/cavity_yz_MALA_000001.jpg` ~ `..._002000.jpg`

2. **1단계 – 객체 탐지 / 분류 (YOLO, AI Hub 기반)**
   - AI Hub에서 제공하는 GPR dataset + 사전 학습된 YOLOv5 모델 사용
   - 클래스: `cavity`, `box`, `patch` 등
   - 결과: `runs/detect/exp*/labels/*.txt` (YOLO 포맷 라벨)

3. **cavity 단면 자동 필터링**
   - `filter_cavity_images.py`
   - YOLO 결과를 읽고, **cavity가 한 번이라도 검출된 이미지의 “원본”** 만
     `classification_cavity_img/` 폴더로 복사

4. **2단계 – 픽셀 단위 세그멘테이션 (U-Net)**
   - 학습용 데이터:  
     - `data2/`: GPR 원본 이미지  
     - `data2_mask/`: 해당 이미지의 cavity 영역을 채운 GT mask (`*_mask.jpg`)
   - `src/train.py`로 U-Net 학습 (BCE + Dice Loss, 간단한 데이터 증강 포함)

5. **cavity 단면에 대한 일괄 mask 생성**
   - `src/batch_inference.py`
   - 입력: `classification_cavity_img/*`  
   - 출력: `classification_cavity_mask/*_mask.png`
   - 나중에 이 mask들을 slice 방향으로 쌓아서 3D cavity volume을 만들 수 있음.

---

## 📂 폴더 구조 (Project Structure)

```text
gpr_to_cavity/
├── continuous_data/                # 연속 GPR 원본(MALA) 이미지 전체
│   ├── cavity_yz_MALA_000001.jpg
│   └── ...
│
├── data2/                          # Segmentation 학습용 GPR 이미지
│   ├── cavity_yz_MALA_000228.jpg
│   └── ...
├── data2_mask/                     # Segmentation 학습용 GT 마스크
│   ├── cavity_yz_MALA_000228_mask.jpg
│   └── ...
│
├── classification_cavity_img/      # YOLO로 cavity가 검출된 원본 단면만 모은 폴더
├── classification_cavity_mask/     # 위 단면들에 대한 U-Net 예측 mask
│
├── checkpoints/
│   └── unet_best.pth               # 현재까지 가장 성능 좋은 U-Net 가중치
│
├── src/
│   ├── dataset.py                  # data / data_mask용 Dataset 클래스
│   ├── model.py                    # Lightweight U-Net 모델 정의
│   ├── train.py                    # U-Net 학습 스크립트 (BCE+Dice, 증강, Scheduler)
│   ├── batch_inference.py          # classification_cavity_img 전체에 대해 mask 예측
│   └── filter_cavity_images.py     # YOLO 결과에서 cavity 이미지만 추출하는 스크립트
│
├── ai_hub/
│   └── src/yolov5_master/          # (외부) AI Hub GPR 탐지 모델 코드 & weights
│       └── runs/detect/exp*/labels # YOLO detection 결과(txt)
│
├── visualization/                  # 예측된 mask들을 3D cavity volume으로 시각화/변환하는 모듈
│   ├── build_cavity_volume.py      # mask들을 쌓아 3D cavity voxel volume 생성
│   ├── view_cavity_volume.py       # PyVista 기반 3D cavity 렌더링
│   ├── view_cavity_slices_spacing.py # slice 간격/보간 실험용 시각화
│   └── cavity_volume.npy           # 생성된 3D voxel cavity 데이터
│
├── slice_interpolation/          # SDT 보간법을 사용하여 이미지 연결
│   ├── build_cavity_volume.py
│   ├── sdt_interpolation.py
│   └── visualize_interp.py 
│
├── outputs/
│   └── prediction_img/             # batch inference로 생성된 예측 mask 이미지 저장 폴더
│
├── note/                           # 실험 과정에서의 메모/기록 파일 모음
│
├── test/                           # 테스트용 코드 및 샘플 이미지/스크립트
│
└── README.md