import os
import shutil

# 현재 스크립트 기준 프로젝트 root 디렉토리(gpr_to_cavity)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(BASE_DIR)    # src/ 기준으로 한 단계 위 이동 → gpr_to_cavity/

# YOLO 탐지 결과 폴더 (bbox 그려진 이미지가 존재)
detect_exp = os.path.join(PROJECT_ROOT, "ai_hub/src/yolov5_master/runs/detect/exp3")
detect_labels = os.path.join(detect_exp, "labels")

# 탐지된 이미지만 저장할 폴더 (gpr_to_cavity 아래 생성)
dst_path = os.path.join(PROJECT_ROOT, "classification_cavity_detection")
os.makedirs(dst_path, exist_ok=True)

# cavity class ID
CAVITY_ID = 1

for label_file in os.listdir(detect_labels):
    if not label_file.endswith(".txt"):
        continue

    txt_path = os.path.join(detect_labels, label_file)

    # txt 내부 라벨 확인
    with open(txt_path, "r") as f:
        lines = f.readlines()

    # cavity 클래스 포함 여부 확인
    has_cavity = any(line.startswith(str(CAVITY_ID)) for line in lines)
    if not has_cavity:
        continue

    # 이미지 파일 이름 = .txt → .jpg
    img_name = label_file.replace(".txt", ".jpg")
    detected_img_path = os.path.join(detect_exp, img_name)

    # 탐지된 이미지 복사
    if os.path.exists(detected_img_path):
        print(f"[SAVE] Detected image copied → {img_name}")
        shutil.copy(detected_img_path, os.path.join(dst_path, img_name))
    else:
        print(f"[MISS] Detection image not found → {img_name}")
