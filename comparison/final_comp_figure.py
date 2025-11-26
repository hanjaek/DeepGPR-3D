import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# 네가 계산한 실제 결과값
# ------------------------
models = ["BCE", "BCE+Aug", "BCE+Dice", "BCE+Dice+Aug"]

dice_scores = [
    0.5336,  # BCE
    0.4572,  # BCE+Aug
    0.6105,  # BCE+Dice
    0.5608   # BCE+Dice+Aug
]

iou_scores = [
    0.4349,  # BCE
    0.3472,  # BCE+Aug
    0.4847,  # BCE+Dice
    0.4555   # BCE+Dice+Aug
]

# ------------------------
# 그래프 설정
# ------------------------
x = np.arange(len(models))
width = 0.35  # 막대 두께

plt.figure(figsize=(10, 6))
plt.title("Comparison of Dice & IoU Scores Across Models", fontsize=14)

# 두 개의 bar 그룹
plt.bar(x - width/2, dice_scores, width, label="Dice", color="#A7D8FF")
plt.bar(x + width/2, iou_scores, width, label="IoU", color="#4A78FF")

# x축 레이블
plt.xticks(x, models, fontsize=12)

# 값 표시
for i in range(len(models)):
    plt.text(x[i] - width/2, dice_scores[i] + 0.01, f"{dice_scores[i]:.2f}", ha='center')
    plt.text(x[i] + width/2, iou_scores[i] + 0.01, f"{iou_scores[i]:.2f}", ha='center')

plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()
