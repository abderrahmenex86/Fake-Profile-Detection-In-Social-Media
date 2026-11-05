from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

frame = pd.read_csv("results/sift_investigation.csv")
order = ["first234", "top256", "bovw256"]
frame["feature"] = pd.Categorical(frame["feature"], categories=order, ordered=True)
figure, axis = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=frame, x="feature", y="test_macro_f1", hue="model", ax=axis)
axis.set_ylim(0, 1)
axis.set_xlabel("SIFT encoding")
axis.set_ylabel("test macro F1")
axis.set_title("SIFT representation investigation")
axis.legend(title="model")
figure.tight_layout()
Path("results/plots").mkdir(parents=True, exist_ok=True)
figure.savefig("results/plots/fig_sift_investigation.png", dpi=180)
plt.close(figure)
print("results/plots/fig_sift_investigation.png")
