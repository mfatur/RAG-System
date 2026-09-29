from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "ragas-evaluation" / "results"
SCORES_FILE = RESULTS_DIR / "ragas_scores.csv"
PLOT_FILE = RESULTS_DIR / "ragas_metrics.png"

scores = pd.read_csv(SCORES_FILE)

metrics = {
    "Faithfulness": "faithfulness",
    "Answer relevancy": "answer_relevancy",
    "Context precision": "context_precision",
    "Context recall": "context_recall",
}

columns = list(metrics.values())

if scores[columns].isna().any().any():
    raise ValueError("Scores contain missing values; refusing to plot incomplete results.")

means = scores[columns].mean()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(metrics.keys(), means, color=["#2878B5", "#55A868", "#C44E52", "#8172B3"])

ax.set_title(f"RAGAS Evaluation (n={len(scores)})")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

for bar, value in zip(bars, means):
    ax.annotate(
        f"{value:.2f}",
        (bar.get_x() + bar.get_width() / 2, value),
        ha="center",
        va="bottom",
    )

fig.tight_layout()
fig.savefig(PLOT_FILE, dpi=160)
print(f"Saved plot: {PLOT_FILE}")