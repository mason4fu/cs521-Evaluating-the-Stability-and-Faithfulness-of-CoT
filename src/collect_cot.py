import os
import re
import time
import json
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset

from utils import OUT, DATA, append_jsonl
from perturb import make_variants

RUNS_PATH = OUT / "runs.jsonl"
USED_QA_PATH = DATA / "used_gsm8k_subset.json"


# ----------------- Utility: extract number -----------------
def extract_number_from_text(text: str) -> str:
    """Extracts the last numeric token from model text."""
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    return matches[-1] if matches else "unknown"


# ----------------- Gemini LLM (with safety + continuation) -----------------
def gemini_cot_and_answer(question: str) -> Dict[str, str]:
    """Generate step-by-step CoT reasoning and numeric answer using Gemini with safety handling and continuation retry."""
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    model = genai.GenerativeModel(model_name)

    base_prompt = (
        "You are a careful math tutor. "
        "These are fictional math problems about numbers, not real people or animals. "
        "Show your reasoning step by step, then end with a line: 'Final Answer: <number>'.\n\n"
        f"Question: {question}\nPlease reason step-by-step."
    )

    def _call_gemini(prompt_text):
        """Single Gemini call with lenient safety settings."""
        return model.generate_content(
            prompt_text,
            generation_config={"temperature": 0.0, "max_output_tokens": 2048},
            safety_settings=[
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
            ],
        )

    try:
        response = _call_gemini(base_prompt)

        # -------- Collect safety info --------
        block_reason = getattr(response.prompt_feedback, "block_reason", None) if hasattr(response, "prompt_feedback") else None
        safety_feedback: List[Dict[str, str]] = []
        finish_reason = None

        # Candidate-level safety info
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            finish_reason = getattr(cand, "finish_reason", None)
            for rating in getattr(cand, "safety_ratings", []):
                cat = getattr(rating.category, "name", str(rating.category))
                prob = getattr(rating.probability, "name", str(rating.probability))
                safety_feedback.append({"category": cat, "probability": prob})

        # Prompt-level safety info
        if hasattr(response, "prompt_feedback") and getattr(response.prompt_feedback, "safety_ratings", None):
            for rating in response.prompt_feedback.safety_ratings:
                cat = getattr(rating.category, "name", str(rating.category))
                prob = getattr(rating.probability, "name", str(rating.probability))
                safety_feedback.append({"category": cat, "probability": prob})

        # -------- Handle blocked/empty responses --------
        if not response.candidates or not response.candidates[0].content.parts:
            raise ValueError(f"Gemini blocked or returned empty response (prompt_block={block_reason}, finish={finish_reason})")

        # Extract text
        text = getattr(response, "text", "") or "".join(
            getattr(p, "text", "") for p in response.candidates[0].content.parts
        ).strip()

        if not text:
            raise ValueError(f"Empty text output from Gemini (finish={finish_reason})")

        # Parse CoT + final answer
        parts = text.split("Final Answer:")
        cot = parts[0].strip()
        final = parts[1].strip() if len(parts) > 1 else extract_number_from_text(text)

        # Safety stop (finish_reason = 2)
        if finish_reason == 2:
            cot += "\n[⚠️ Stopped early due to Gemini safety filtering.]"

        # Normal completion
        if finish_reason == 1:
            return {
                "cot": cot,
                "final_answer": final,
                "safety_feedback": safety_feedback,
                "prompt_block_reason": block_reason,
                "finish_reason": finish_reason,
            }

        # -------- Continue generation on ERROR --------
        if finish_reason == "ERROR" or not final or final == "unknown":
            print(f"⚠️ Gemini stopped early (finish_reason={finish_reason}). Asking model to continue...")

            continuation_prompt = (
                f"The reasoning so far was:\n{cot}\n\n"
                "Please continue reasoning and complete your solution. "
                "If you already know the final numeric answer, end with 'Final Answer: <number>'."
            )

            retry_response = _call_gemini(continuation_prompt)
            retry_text = getattr(retry_response, "text", "") or "".join(
                getattr(p, "text", "") for p in retry_response.candidates[0].content.parts
            ).strip()

            parts = retry_text.split("Final Answer:")
            retry_cot = parts[0].strip()
            retry_final = parts[1].strip() if len(parts) > 1 else extract_number_from_text(retry_text)

            return {
                "cot": cot + "\n[Continued generation]\n" + retry_cot,
                "final_answer": retry_final,
                "safety_feedback": safety_feedback,
                "prompt_block_reason": block_reason,
                "finish_reason": "CONTINUED",
            }

        return {
            "cot": cot,
            "final_answer": final,
            "safety_feedback": safety_feedback,
            "prompt_block_reason": block_reason,
            "finish_reason": finish_reason,
        }

    except Exception as e:
        return {
            "cot": f"[BLOCKED] Gemini blocked or errored for question: {e}",
            "final_answer": "unknown",
            "safety_feedback": [],
            "prompt_block_reason": str(e),
            "finish_reason": "ERROR",
        }


# ----------------- Main Runner -----------------
def run():
    if RUNS_PATH.exists():
        RUNS_PATH.unlink()

    # Load small test subset (edit here to pick specific questions)
    ds = load_dataset("gsm8k", "main")["train"]
    dataset = ds.shuffle(seed=44).select(range(3)).to_list()

    # Save the used subset
    USED_QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USED_QA_PATH, "w") as f:
        json.dump(
            [{"id": ex.get("id", f"q{idx+1}"), "question": ex["question"], "answer": ex["answer"]}
             for idx, ex in enumerate(dataset)],
            f, indent=2
        )

    print(f"\nLoaded {len(dataset)} GSM8K questions.")
    print(f"Subset saved to: {USED_QA_PATH}\n")
    print("Running CoT collection with: Gemini\n")

    for idx, ex in enumerate(dataset):
        q_id = ex.get("id", f"q{idx+1}")
        q_text = ex["question"]

        for v in make_variants({"question": q_text}):
            out = gemini_cot_and_answer(v["text"])
            record = {
                "id": q_id,
                "gold_answer": ex["answer"],
                "variant": v["variant"],
                "question_text": v["text"],
                "cot": out["cot"],
                "final_answer": out["final_answer"],
                "safety_feedback": out.get("safety_feedback", []),
                "prompt_block_reason": out.get("prompt_block_reason"),
                "finish_reason": out.get("finish_reason"),
                "ts": time.time(),
            }
            append_jsonl(RUNS_PATH, record)

    print(f"\n✅ Done. Saved {len(dataset)} questions × 4 variants to {RUNS_PATH}")


if __name__ == "__main__":
    run()
