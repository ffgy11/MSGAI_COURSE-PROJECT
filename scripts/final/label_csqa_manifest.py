#!/usr/bin/env python3
"""Add objective CommonsenseQA correctness/hallucination labels to a manifest."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

LETTERS = "ABCDE"

def parse_letter(text: str | None) -> str | None:
    if not text:
        return None
    s = str(text).strip()
    patterns = [
        r"(?i)\banswer\s*(?:is|:)?\s*\(?([A-E])\)?\b",
        r"(?i)^\s*\(?([A-E])\)?(?:[\s\.\:\-]|$)",
        r"(?i)\b(?:option|choice)\s*\(?([A-E])\)?\b",
        r"(?i)\b([A-E])\b",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1).upper()
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = parsed = correct = 0
    with inp.open("r", encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            gold = str(row.get("answer_key", row.get("answerKey", ""))).strip().upper()
            pred = parse_letter(row.get("generation", row.get("generated_text", row.get("response"))))
            if gold not in LETTERS:
                row["hallucination"] = None
            elif pred is None:
                row["hallucination"] = {"consistent": False, "label": "hallucination", "reason": "unparsed_mc_answer"}
            else:
                parsed += 1
                ok = pred == gold
                correct += int(ok)
                row["prediction_letter"] = pred
                row["hallucination"] = {
                    "consistent": bool(ok),
                    "label": "non_hallucination" if ok else "hallucination",
                    "source": "commonsenseqa_ground_truth",
                }
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"rows={n} parsed={parsed} correct={correct} accuracy={correct/max(parsed,1):.4f}")
    print(f"saved={out}")

if __name__ == "__main__":
    main()
