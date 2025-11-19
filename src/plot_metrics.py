import matplotlib.pyplot as plt

metrics = ["Dice", "IoU", "PixelAcc"]
values = [0.5608, 0.4555, 0.9814]

colors = ["#3b82f6", "#6366f1", "#06b6d4"]  # 파란색 계열

plt.figure(figsize=(7, 4))
plt.bar(metrics, values, color=colors)

plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Lightweight U-Net Segmentation Performance")

for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("unet_metrics_colored.png", dpi=300)
plt.show()
