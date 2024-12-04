import pandas as pd
import matplotlib.pyplot as plt

deviations_df = pd.read_csv("data/feature_deviations.csv")

features = [
    "spectral_centroid_mean_deviation",
    "spectral_bandwidth_mean_deviation",
    "tempo_deviation",
    "spectral_contrast_mean_deviation",
]

plt.figure(figsize=(12, 8))
for feature in features:
    plt.plot(
        deviations_df.index,
        deviations_df[feature],
        marker="o",
        label=feature.replace("_deviation", "").replace("_", " ").title()
    )

plt.title("Feature Deviations Over Training Epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Mean Deviation from Target", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()

plt.savefig("feature_deviations_plot.png")
plt.show()
print("Feature deviation plot saved as 'feature_deviations_plot.png'.")
