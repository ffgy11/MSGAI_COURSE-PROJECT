#!/usr/bin/env python3
"""Merge externally prepared binary hallucination labels into a trajectory manifest.

CSV must contain one identifier column (index or question_id) and a binary label column.
Use this for TriviaQA/HotpotQA when labels come from an external or prior judge pipeline.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def as_label(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1","true","hallucination","hallucinated","hall"}: return 1
        if s in {"0","false","non_hallucination","non-hallucination","factual","correct"}: return 0
    try:
        x = int(v)
        if x in (0,1): return x
    except Exception:
        pass
    raise ValueError(f"unsupported label value: {v!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--id-column", default="index", choices=["index","question_id"])
    ap.add_argument("--label-column", default="label")
    args = ap.parse_args()
    lab = pd.read_csv(args.labels)
    if args.id_column not in lab or args.label_column not in lab:
        raise ValueError(f"labels CSV must contain {args.id_column!r} and {args.label_column!r}")
    mapping = {str(r[args.id_column]): as_label(r[args.label_column]) for _, r in lab.iterrows()}
    inp, out = Path(args.manifest), Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    matched = total = 0
    with inp.open("r", encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip(): continue
            row = json.loads(line); total += 1
            key = str(row.get(args.id_column, ""))
            if key in mapping:
                y = mapping[key]; matched += 1
                row["hallucination"] = {
                    "consistent": bool(y == 0),
                    "label": "hallucination" if y else "non_hallucination",
                    "source": "external_label_merge",
                }
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"matched={matched}/{total}")
    print(f"saved={out}")

if __name__ == "__main__": main()
