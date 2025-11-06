import pandas as pd
import matplotlib.pyplot as plt
from utils import OUT, FIG
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

def main():
    summary = pd.read_csv(OUT / "summary.csv")

    # Histogram: TF-IDF cosine by variant
    for metric in ["cosine_tfidf", "edit_sim", "cosine_embed", "change_rate"]:
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

    # Bar plot of accuracy
    plt.figure(figsize=(6,4))
    plt.bar(acc["variant"], acc["is_correct"], color="skyblue")
    plt.title("Accuracy by Variant")
    plt.ylabel("Proportion Correct")
    plt.xlabel("Variant")
    plt.ylim(0, 1)
    plt.savefig(FIG / "accuracy_bar.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG / 'accuracy_bar.png'}")
    
    #Scatter of Semantic similarity vs correctness
    if "cosine_embed" in summary.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(summary["cosine_embed"], summary["is_correct"], alpha=0.5)
        plt.xlabel("Semantic Similarity (cosine_embed)")
        plt.ylabel("Correct (1/0)")
        plt.title("Semantic Similarity vs Correctness")
        plt.savefig(FIG / "similarity_vs_accuracy.png", bbox_inches="tight")
        plt.close()
        print(f"Saved: {FIG / 'similarity_vs_accuracy.png'}")
if __name__ == "__main__":
    main()