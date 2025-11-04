import re
from typing import List, Dict

def lexical_variant(text: str) -> str:
    # simple synonym replacements
    repl = {
        r"\bbought\b": "purchased",
        r"\bholds\b": "contains",
        r"\btota[l|l]\b": "overall",
        r"\bin all\b": "in total"
    }
    out = text
    for k,v in repl.items():
        out = re.sub(k, v, out, flags=re.IGNORECASE)
    return out

def syntactic_variant(text: str) -> str:
    # naive clause reorder: move last sentence phrase to front if possible
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        return parts[-1].capitalize() + ", " + ", ".join(parts[:-1]) + "."
    return text.replace("How many", "Compute how many")

def pragmatic_variant(text: str) -> str:
    if text.endswith("?"):
        return "Please answer briefly: " + text
    return "Please answer briefly: " + text + "?"

def make_variants(q: Dict) -> List[Dict]:
    base = q["question"].strip()
    return [
        {"variant":"original",  "text": base},
        {"variant":"lexical",   "text": lexical_variant(base)},
        {"variant":"syntactic", "text": syntactic_variant(base)},
        {"variant":"pragmatic", "text": pragmatic_variant(base)},
    ]