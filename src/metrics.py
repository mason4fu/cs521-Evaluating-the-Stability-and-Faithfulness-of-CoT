import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance as td
import reimport difflib

from sentence_transformers import SentenceTransformer, util
model_embed = SentenceTransformer('all-MiniLM-L6-v2')

from utils import OUT

RUNS_PATH = OUT / "runs.jsonl"
SUMMARY = OUT / "summary.csv"


def load_runs():
    rows = []
    with open(RUNS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def extract_gold_number(gold_text: str) -> str:
    """Extract final numeric answer from GSM8K gold answer text."""
    matches = re.findall(r"####\s*([-+]?\d*\.?\d+)", gold_text)
    return matches[-1] if matches else "unknown"


def normalize_num(s: str):
    """Normalize numeric strings to comparable format."""
    try:
        return str(int(float(s.strip())))
    except Exception:
        return s.strip().lower()


def change_rate(a,b):
    return 1 - difflib.SequenceMatcher(None,a,b).ratio()

def compute_pairwise(df):
    recs = []
    for qid in df["id"].unique():
        sub = df[df["id"] == qid]
        base = sub[sub["variant"] == "original"].iloc[0]
        base_cot = base["cot"]
        base_ans = normalize_num(base["final_answer"])
        gold = extract_gold_number(base["gold_answer"])

        for _, row in sub.iterrows():
            var = row["variant"]
            cot = row["cot"]
            ans = normalize_num(row["final_answer"])

            # TF-IDF cosine
            vect = TfidfVectorizer().fit([base_cot, cot])
            cos = float(cosine_similarity(
                vect.transform([base_cot]),
                vect.transform([cot])
            )[0,0])
            A_emb = model_embed.encode(base_cot)
            B_emb = model_embed.encode(cot)
            sem_cos = float(util.cos_sim(A_emb,B_emb))
            
            chg = change_rate(base_cot, cot)
            cos = float(cosine_similarity(vect.transform([base_cot]), vect.transform([cot]))[0, 0])

            # Edit similarity
            lev = 1.0 - td.levenshtein.normalized_distance(base_cot, cot)

            correct = int(ans == gold)

            recs.append({
                "id": qid,
                "variant": var,
                "cosine_tfidf": round(cos, 4),
                "edit_sim": round(lev, 4),
                "cosine_embed": round(sem_cos,4),
                "change_rate": round(chg,4),
                "answer": ans,
                "gold": gold,
                "is_correct": correct,
                "model": row.get("model","unknown")
            })
    return pd.DataFrame(recs)


def main():
    df = load_runs()
    out = compute_pairwise(df)
    out.to_csv(SUMMARY, index=False)
    print(f"Saved: {SUMMARY}")


if __name__ == "__main__":
    main()
