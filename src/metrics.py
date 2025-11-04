import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance as td

from utils import OUT

RUNS_PATH = OUT / "runs.jsonl"
SUMMARY = OUT / "summary.csv"

def load_runs():
    rows = []
    with open(RUNS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def normalize_num(s: str):
    try:
        return str(int(float(s)))
    except:
        return s.strip()

def compute_pairwise(df):
    # compare each variant to the "original" for same id
    recs = []
    ids = df["id"].unique()
    for qid in ids:
        sub = df[df["id"] == qid]
        base = sub[sub["variant"] == "original"].iloc[0]
        base_cot = base["cot"]
        base_ans = normalize_num(base["final_answer"])
        gold = normalize_num(base["gold_answer"])

        for _, row in sub.iterrows():
            var = row["variant"]
            cot = row["cot"]
            ans = normalize_num(row["final_answer"])

            # TF-IDF cosine (semantic-ish for small demo)
            vect = TfidfVectorizer().fit([base_cot, cot])
            A = vect.transform([base_cot])
            B = vect.transform([cot])
            cos = float(cosine_similarity(A, B)[0,0])

            # normalized edit similarity (1 - normalized Levenshtein distance)
            lev = 1.0 - td.levenshtein.normalized_distance(base_cot, cot)

            correct = (ans == gold)

            recs.append({
                "id": qid,
                "variant": var,
                "cosine_tfidf": round(cos, 4),
                "edit_sim": round(lev, 4),
                "answer": ans,
                "gold": gold,
                "is_correct": int(correct)
            })
    return pd.DataFrame(recs)

def main():
    df = load_runs()
    out = compute_pairwise(df)
    out.to_csv(SUMMARY, index=False)
    print(f"Saved: {SUMMARY}")

if __name__ == "__main__":
    main()