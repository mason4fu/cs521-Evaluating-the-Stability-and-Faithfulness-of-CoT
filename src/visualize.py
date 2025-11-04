import pandas as pd
import matplotlib.pyplot as plt
from utils import OUT, FIG

def main():
    summary = pd.read_csv(OUT / "summary.csv")

    # Histogram: TF-IDF cosine by variant
    for metric in ["cosine_tfidf", "edit_sim"]:
        plt.figure()
        for var, sub in summary.groupby("variant"):
            sub[metric].plot(kind="hist", alpha=0.5, bins=10, label=var)
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.legend()
        figpath = FIG / f"{metric}_hist.png"
        plt.savefig(figpath, bbox_inches="tight")
        plt.close()
        print(f"Saved: {figpath}")

    # Accuracy by variant
    acc = summary.groupby("variant")["is_correct"].mean().reset_index()
    acc.to_csv(OUT / "accuracy_by_variant.csv", index=False)
    print(f"Saved: {OUT / 'accuracy_by_variant.csv'}")

if __name__ == "__main__":
    main()