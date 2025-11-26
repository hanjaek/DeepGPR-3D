import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# Scores (네가 측정한 값)
# ------------------------
models = ["BCE", "BCE+Dice"]

dice_scores = [
    0.5336,  # BCE
    0.6105   # BCE+Dice
]

iou_scores = [
    0.4349,  # BCE
    0.4847   # BCE+Dice
]

# ------------------------
# Graph settings
# ------------------------
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(8,5))
plt.title("Comparison of Dice & IoU Scores Across Models", fontsize=14)

# Colors (파스텔)
color_dice = "#A7D8FF"   # 밝은 하늘 파스텔
color_iou  = "#4A78FF"   # 진한 파란 파스텔

# Bars
plt.bar(x - width/2, dice_scores, width, label="Dice", color=color_dice)
plt.bar(x + width/2, iou_scores,  width, label="IoU",  color=color_iou)

# X-axis labels
plt.xticks(x, models, fontsize=12)

# Value labels
for i in range(len(models)):
    plt.text(x[i] - width/2, dice_scores[i] + 0.01, f"{dice_scores[i]:.3f}", ha='center')
    plt.text(x[i] + width/2, iou_scores[i]  + 0.01, f"{iou_scores[i]:.3f}", ha='center')

plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()
