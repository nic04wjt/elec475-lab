"""Builds plots/ablation_summary.png from the train.py epoch logs (CIFAR-10 runs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {
    "LeNet-5 (5 ep, CPU)":          ([46.28, 49.99, 50.36, 52.74, 52.98],
                                     [1.5042, 1.3984, 1.3934, 1.3337, 1.3154]),
    "Modernized LeNet (5 ep, CPU)": ([52.06, 59.44, 62.45, 64.73, 64.51],
                                     [1.3164, 1.1379, 1.0655, 1.0166, 1.0118]),
    "AlexNet, no norm (10 ep)":     ([46.38, 57.09, 60.20, 63.94, 66.26, 67.05, 70.14, 71.40, 71.46, 73.52],
                                     [1.4382, 1.2065, 1.1117, 1.0241, 0.9689, 0.9517, 0.8765, 0.8228, 0.8187, 0.7717]),
    "AlexNet + LRN (10 ep)":        ([44.45, 56.59, 58.86, 64.37, 68.95, 69.54, 73.01, 72.11, 75.10, 75.14],
                                     [1.4718, 1.1929, 1.1661, 1.0141, 0.9054, 0.8710, 0.7973, 0.8217, 0.7438, 0.7452]),
    "AlexNet + BN (15 ep)":         ([52.56, 60.76, 64.29, 70.89, 73.79, 68.76, 76.40, 79.62, 79.53, 79.93,
                                      80.53, 79.17, 78.54, 81.18, 78.59],
                                     [1.3344, 1.1341, 1.0133, 0.8424, 0.7672, 0.9240, 0.7287, 0.6105, 0.6178, 0.6172,
                                      0.5933, 0.6511, 0.7193, 0.6079, 0.7104]),
    "AlexNet + BN + Aug (15 ep)":   ([50.95, 59.00, 66.69, 68.60, 68.98, 69.62, 76.86, 76.89, 75.94, 78.73,
                                      79.94, 80.81, 82.83, 79.60, 81.65],
                                     [1.4207, 1.1831, 0.9362, 0.8940, 0.9485, 0.9125, 0.6952, 0.6891, 0.7217, 0.6593,
                                      0.5913, 0.5887, 0.5181, 0.6185, 0.5222]),
}
colors = ["#9e9e9e", "#616161", "#f4a259", "#8cb369", "#3a86ff", "#d62828"]

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
for (name, (acc, loss)), c in zip(runs.items(), colors):
    epochs = range(1, len(acc) + 1)
    ax[0].plot(epochs, acc, marker="o", ms=4, color=c, label=name)
    ax[1].plot(epochs, loss, marker="s", ms=4, color=c, label=name)
ax[0].set(title="Validation accuracy (CIFAR-10)", xlabel="Epoch", ylabel="Accuracy (%)")
ax[1].set(title="Validation loss (CIFAR-10)", xlabel="Epoch", ylabel="Cross-entropy loss")
for a in ax[:2]:
    a.grid(ls="--", alpha=0.5)
    a.legend(fontsize=8)

names = list(runs)
best = [max(acc) for acc, _ in runs.values()]
bars = ax[2].barh(names, best, color=colors)
for bar, v in zip(bars, best):
    ax[2].text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", fontsize=9)
ax[2].set(title="Best validation accuracy (CIFAR-10)", xlabel="Accuracy (%)", xlim=(0, 95))
ax[2].invert_yaxis()
ax[2].grid(axis="x", ls="--", alpha=0.5)

fig.suptitle("ELEC 475 Lab 2 - Master Ablation Summary", fontsize=14)
fig.tight_layout()
fig.savefig("plots/ablation_summary.png", dpi=150)
print("saved plots/ablation_summary.png")
