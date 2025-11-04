import os, re, time
from pathlib import Path
from typing import Dict
from utils import DATA, OUT, append_jsonl, load_json, get_flag
from perturb import make_variants

RUNS_PATH = OUT / "runs.jsonl"

USE_OPENAI = get_flag("USE_OPENAI", False)
USE_GEMINI = get_flag("USE_GEMINI", False)

OPENAI_OK = bool(os.environ.get("OPENAI_API_KEY"))
GEMINI_OK = bool(os.environ.get("GEMINI_API_KEY"))

# ----------------- Toy Reasoner (offline) -----------------
def toy_cot_and_answer(question: str) -> Dict[str,str]:
    """
    Minimal rule-based CoT: tries to parse +/x counts and show steps.
    Only meant to generate *some* CoT text to test the pipeline.
    """
    q = question
    # detect two integers (very naive)
    nums = list(map(int, re.findall(r"\b\d+\b", q)))
    cot = "Let's think step by step.\n"
    ans = None

    if "box" in q.lower() and ("hold" in q.lower() or "contain" in q.lower()):
        # multiplication pattern: X per box * Y boxes
        if len(nums) >= 2:
            per, boxes = nums[0], nums[1]
            cot += f"There are {per} per box and {boxes} boxes.\n"
            cot += f"Multiply: {per} * {boxes} = {per*boxes}.\n"
            ans = per * boxes
    elif len(nums) >= 2 and any(k in q.lower() for k in ["in all","in total","overall","now","total"]):
        a, b = nums[0], nums[1]
        cot += f"Add the amounts: {a} + {b} = {a+b}.\n"
        ans = a + b
    elif len(nums) >= 2:
        a, b = nums[0], nums[1]
        cot += f"Assume addition unless specified: {a} + {b} = {a+b}.\n"
        ans = a + b
    else:
        cot += "No numbers found; cannot compute.\n"
        ans = "unknown"

    cot += "Therefore, the final answer is computed above."
    return {"cot": cot, "final_answer": str(ans)}

# ----------------- OpenAI LLM (optional) ------------------
def openai_cot_and_answer(question: str) -> Dict[str,str]:
    from openai import OpenAI
    client = OpenAI()
    sys = "You are a careful math tutor. Think step by step before giving the final answer."
    prompt = f"Question: {question}\nShow your reasoning, then give Final Answer: <number>."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":prompt}],
        temperature=0
    )
    text = resp.choices[0].message.content
    # crude split for demo
    parts = text.split("Final Answer:")
    cot = parts[0].strip()
    final = parts[1].strip() if len(parts) > 1 else "unknown"
    return {"cot": cot, "final_answer": final}

# ----------------- Gemini LLM (optional) ------------------
def gemini_cot_and_answer(question: str) -> Dict[str, str]:
    """
    Calls Google's Gemini to produce a step-by-step CoT and a final answer line.
    Expects GEMINI_API_KEY in environment.
    """
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    # Choose a reasoning-capable or standard model
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    model = genai.GenerativeModel(model_name)

    system = (
        "You are a careful math tutor. Show step-by-step reasoning (chain of thought), "
        "then end with a line starting exactly with: Final Answer: <number>"
    )
    user = f"Question: {question}\nPlease reason step-by-step, then provide 'Final Answer: <number>'."

    # Keep outputs deterministic/short
    response = model.generate_content(
        [system, user],
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 512,
        }
    )

    text = response.text or ""
    # crude split for demo compatibility with the existing pipeline
    parts = text.split("Final Answer:")
    cot = parts[0].strip()
    final = parts[1].strip() if len(parts) > 1 else "unknown"
    return {"cot": cot, "final_answer": final}


def run():
    if RUNS_PATH.exists():
        RUNS_PATH.unlink()

    dataset = load_json(DATA / "gsm8k_sample.json")

    # priority: Gemini > OpenAI > toy, based on flags and key presence
    use_gemini = USE_GEMINI and GEMINI_OK
    use_openai = (not use_gemini) and USE_OPENAI and OPENAI_OK

    for ex in dataset:
        for v in make_variants(ex):
            if use_gemini:
                out = gemini_cot_and_answer(v["text"])
            elif use_openai:
                out = openai_cot_and_answer(v["text"])
            else:
                out = toy_cot_and_answer(v["text"])

            record = {
                "id": ex["id"],
                "gold_answer": ex["answer"],
                "variant": v["variant"],
                "question_text": v["text"],
                "cot": out["cot"],
                "final_answer": out["final_answer"],
                "ts": time.time()
            }
            append_jsonl(RUNS_PATH, record)

if __name__ == "__main__":
    run()