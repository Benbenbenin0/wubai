import pandas as pd
import matplotlib.pyplot as plt

dtw_df = pd.read_csv("data/dtw_costs.csv")

plt.figure(figsize=(10, 6))
plt.plot(dtw_df["epoch"], dtw_df["dtw_cost"], marker="o", label="DTW Cost")

plt.title("Melody Alignment (DTW Costs) Over Training Epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Average DTW Cost", fontsize=14)
plt.grid(alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("dtw_costs_plot.png")
plt.show()
print("DTW cost plot saved as 'dtw_costs_plot.png'.")
