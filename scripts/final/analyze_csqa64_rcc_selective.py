#!/usr/bin/env python3
"""
Offline CSQA-64 RCC selective mitigation from existing caches.

No model loading. No new diffusion generation. No LLM judge.

The script:
  1) auto-discovers the LLaDA-64 CommonsenseQA manifest;
  2) auto-discovers RCC per-sample scores for LLaDA-64 / CommonsenseQA;
  3) reconstructs ground-truth correctness from the manifest;
  4) evaluates RCC-guided abstention at multiple coverage levels;
  5) writes a summary, CSV, and risk-coverage plot;
  6) if a CSQA-128 results CSV is available, also writes a 64-vs-128 comparison.

Expected RCC score semantics:
  higher `score__pcs_plus_selection` = higher hallucination/error risk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LETTERS = "ABCDE"


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

def score_candidate(path: Path) -> int:
    s = str(path).lower()
    score = 0
    if "commonsense" in s or "csqa" in s:
        score += 10
    if "64" in s:
        score += 10
    if "rcc" in s:
        score += 5
    if "sample_scores" in path.name.lower():
        score += 5
    if "128" in s:
        score -= 20
    return score


def manifest_candidate(path: Path) -> int:
    s = str(path).lower()
    score = 0
    if "commonsense" in s or "csqa" in s:
        score += 10
    if "64" in s:
        score += 10
    if "manifest" in path.name.lower():
        score += 5
    if "128" in s:
        score -= 20
    return score


def discover_file(root: Path, patterns: list[str], scorer, required_score: int = 10) -> Optional[Path]:
    found = []
    for pat in patterns:
        found.extend(root.glob(pat))
    found = [p for p in found if p.is_file()]
    if not found:
        return None
    found = sorted(found, key=lambda p: (scorer(p), -len(str(p))), reverse=True)
    if scorer(found[0]) < required_score:
        return None
    return found[0]


# -----------------------------------------------------------------------------
# Generic parsing
# -----------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recursive_find(obj: Any, keys: Iterable[str]) -> Any:
    wanted = {str(k).lower() for k in keys}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in wanted and v is not None:
                return v
        for v in obj.values():
            hit = recursive_find(v, keys)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = recursive_find(v, keys)
            if hit is not None:
                return hit
    return None


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip().lower()


def extract_question(row: dict) -> Optional[str]:
    x = recursive_find(row, ["question", "question_text", "query", "input_question"])
    return str(x).strip() if isinstance(x, str) and x.strip() else None


def extract_choices(row: dict):
    obj = recursive_find(row, ["choices", "options", "answer_choices"])
    if isinstance(obj, dict):
        labels = obj.get("label", obj.get("labels"))
        texts = obj.get("text", obj.get("texts", obj.get("choices")))
        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            return [str(x).upper() for x in labels], [str(x) for x in texts]
    if isinstance(obj, list) and obj and all(isinstance(x, str) for x in obj):
        return list(LETTERS[:len(obj)]), list(obj)

    # A/B/C/D/E as dict keys.
    for container in [row] + [v for v in row.values() if isinstance(v, dict)]:
        vals = []
        labs = []
        for c in LETTERS:
            if c in container:
                labs.append(c)
                vals.append(str(container[c]))
            elif c.lower() in container:
                labs.append(c)
                vals.append(str(container[c.lower()]))
            else:
                break
        if len(vals) >= 2:
            return labs, vals
    return None


def extract_gold(row: dict, n_choices: int | None = None) -> Optional[str]:
    v = recursive_find(
        row,
        [
            "answerKey",
            "answer_key",
            "gold_answer_key",
            "correct_option",
            "correct_choice",
            "gold_letter",
        ],
    )
    if isinstance(v, str):
        s = v.strip().upper()
        if s in LETTERS:
            return s
        if s.isdigit():
            z = int(s)
            if n_choices:
                if 0 <= z < n_choices:
                    return LETTERS[z]
                if 1 <= z <= n_choices:
                    return LETTERS[z - 1]
    if isinstance(v, (int, np.integer)) and n_choices:
        z = int(v)
        if 0 <= z < n_choices:
            return LETTERS[z]
        if 1 <= z <= n_choices:
            return LETTERS[z - 1]
    return None


BASELINE_TEXT_KEYS = [
    "generated_text",
    "generation",
    "model_output",
    "response",
    "model_response",
    "prediction_text",
    "decoded_output",
    "output_text",
]


def extract_baseline_text(row: dict) -> Optional[str]:
    v = recursive_find(row, BASELINE_TEXT_KEYS)
    return str(v).strip() if isinstance(v, str) and v.strip() else None


def parse_letter(text: Optional[str], labels: list[str], choices: list[str]) -> Optional[str]:
    if not text:
        return None
    s = str(text).strip()
    for pat in [
        r"(?i)\banswer\s*(?:is|:)?\s*\(?([A-E])\)?\b",
        r"(?i)^\s*\(?([A-E])\)?(?:[\s\.\:\-]|$)",
        r"(?i)\b(?:option|choice)\s*\(?([A-E])\)?\b",
        r"(?i)\b([A-E])\b",
    ]:
        m = re.search(pat, s)
        if m and m.group(1).upper() in labels:
            return m.group(1).upper()

    ns = norm_text(s)
    hits = []
    for lab, ch in zip(labels, choices):
        nc = norm_text(ch)
        if nc and (ns == nc or nc in ns):
            hits.append(lab)
    return hits[0] if len(hits) == 1 else None


# -----------------------------------------------------------------------------
# Optional HF fallback
# -----------------------------------------------------------------------------

def load_hf_lookup():
    try:
        from datasets import load_dataset
    except Exception:
        return {}

    for name in ["tau/commonsense_qa", "commonsense_qa"]:
        try:
            ds = load_dataset(name, split="validation")
            out = {}
            for r in ds:
                out[norm_text(r["question"])] = dict(r)
            return out
        except Exception:
            pass
    return {}


def enrich_manifest(rows: list[dict]) -> list[dict]:
    prelim = []
    need_hf = False

    for i, row in enumerate(rows):
        q = extract_question(row)
        ch = extract_choices(row)
        gold = extract_gold(row, len(ch[1]) if ch else None)
        txt = extract_baseline_text(row)

        if q is None or ch is None or gold is None:
            need_hf = True

        prelim.append({
            "manifest_index": i,
            "raw": row,
            "question": q,
            "choice_pair": ch,
            "gold": gold,
            "baseline_text": txt,
        })

    lookup = load_hf_lookup() if need_hf else {}

    out = []
    for x in prelim:
        q = x["question"]
        ch = x["choice_pair"]
        gold = x["gold"]

        if q is None:
            raise RuntimeError(
                f"Manifest row {x['manifest_index']} has no question text."
            )

        if ch is None or gold is None:
            match = lookup.get(norm_text(q))
            if match is None:
                raise RuntimeError(
                    "Manifest lacks choices/gold and Hugging Face fallback could not "
                    f"match question:\n{q}"
                )
            if ch is None:
                ch = (
                    [str(v).upper() for v in match["choices"]["label"]],
                    [str(v) for v in match["choices"]["text"]],
                )
            if gold is None:
                gold = str(match["answerKey"]).strip().upper()

        labels, choices = ch
        raw = x["raw"]
        idx = raw.get("index", raw.get("question_id", x["manifest_index"]))
        entropy_file = raw.get("entropy_file")

        pred = parse_letter(x["baseline_text"], labels, choices)

        out.append({
            "manifest_index": x["manifest_index"],
            "index": idx,
            "question": q,
            "gold": gold,
            "baseline_text": x["baseline_text"],
            "baseline_letter": pred,
            "baseline_correct": None if pred is None else bool(pred == gold),
            "entropy_file": entropy_file,
        })
    return out


# -----------------------------------------------------------------------------
# RCC score matching
# -----------------------------------------------------------------------------

def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    candidates = [
        "score__pcs_plus_selection",
        "pcs_plus_selection",
        "rcc_score",
        "score",
    ]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise RuntimeError(
            f"No RCC score column found in {path}. Columns: {list(df.columns)}"
        )
    df = df.copy()
    df["_rcc_score"] = pd.to_numeric(df[col], errors="coerce")
    return df


def attach_scores(rows: list[dict], scores: pd.DataFrame) -> None:
    by_id = {}
    for c in ["index", "question_id", "manifest_index"]:
        if c in scores.columns:
            for _, r in scores[[c, "_rcc_score"]].dropna().iterrows():
                by_id[str(r[c])] = float(r["_rcc_score"])

    by_entropy = {}
    if "entropy_file" in scores.columns:
        for _, r in scores[["entropy_file", "_rcc_score"]].dropna().iterrows():
            by_entropy[Path(str(r["entropy_file"])).name] = float(r["_rcc_score"])

    for row in rows:
        val = None
        for key in [row["index"], row["manifest_index"]]:
            if str(key) in by_id:
                val = by_id[str(key)]
                break
        if val is None and row.get("entropy_file"):
            val = by_entropy.get(Path(str(row["entropy_file"])).name)
        if val is not None:
            row["rcc_score"] = val

    # Row-order fallback if dimensions line up exactly.
    if any("rcc_score" not in r for r in rows) and len(scores) == len(rows):
        vals = scores["_rcc_score"].to_numpy(dtype=float)
        for i, row in enumerate(rows):
            if "rcc_score" not in row and np.isfinite(vals[i]):
                row["rcc_score"] = float(vals[i])

    missing = [r for r in rows if "rcc_score" not in r or not np.isfinite(r["rcc_score"])]
    if missing:
        raise RuntimeError(
            f"Could not match RCC scores for {len(missing)}/{len(rows)} rows."
        )


# -----------------------------------------------------------------------------
# Selective answering
# -----------------------------------------------------------------------------

def selective_curve(df: pd.DataFrame) -> pd.DataFrame:
    correct = df["baseline_correct"].astype(bool).to_numpy()
    risk = df["rcc_score"].astype(float).to_numpy()

    baseline_acc = float(correct.mean())
    baseline_err = 1.0 - baseline_acc

    rows = []
    for abstain_rate in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        if abstain_rate == 0:
            keep = np.ones(len(df), dtype=bool)
            threshold = float("inf")
        else:
            threshold = float(np.quantile(risk, 1.0 - abstain_rate))
            keep = risk < threshold

        ans_acc = float(correct[keep].mean())
        ans_err = 1.0 - ans_acc
        rej_acc = float(correct[~keep].mean()) if (~keep).any() else np.nan

        rel_red = (
            (baseline_err - ans_err) / baseline_err
            if baseline_err > 0 else np.nan
        )

        rows.append({
            "abstain_rate": abstain_rate,
            "coverage": float(keep.mean()),
            "answered_n": int(keep.sum()),
            "abstained_n": int((~keep).sum()),
            "risk_threshold": threshold,
            "answered_accuracy": ans_acc,
            "answered_error_rate": ans_err,
            "relative_error_reduction": rel_red,
            "abstained_subset_accuracy": rej_acc,
            "abstained_subset_error_rate": (
                1.0 - rej_acc if np.isfinite(rej_acc) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--rcc_scores", default=None)
    ap.add_argument(
        "--reference128",
        default="outputs/rcc_verify_csqa/results.csv",
        help="Optional 128-step per-sample results containing rcc_score/baseline_correct.",
    )
    ap.add_argument(
        "--output_dir",
        default="outputs/rcc_selective_csqa64",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest) if args.manifest else discover_file(
        root,
        ["**/manifest.jsonl", "**/*manifest*.jsonl"],
        manifest_candidate,
        required_score=15,
    )
    score_path = Path(args.rcc_scores) if args.rcc_scores else discover_file(
        root,
        ["**/sample_scores.csv", "**/*score*.csv"],
        score_candidate,
        required_score=15,
    )

    if manifest is None:
        print("ERROR: could not auto-discover a CSQA-64 manifest.")
        print("Try: --manifest /path/to/commonsenseqa_steps64_400/manifest.jsonl")
        raise SystemExit(2)

    if score_path is None:
        print("ERROR: could not auto-discover CSQA-64 RCC sample scores.")
        print("Try: --rcc_scores /path/to/llada_steps64/commonsenseqa/sample_scores.csv")
        raise SystemExit(2)

    print("=" * 108)
    print("OFFLINE CSQA-64 RCC SELECTIVE MITIGATION")
    print("=" * 108)
    print(f"manifest   : {manifest}")
    print(f"RCC scores : {score_path}")
    print("model load : NO")
    print("generation : NO")
    print("LLM judge  : NO")
    print()

    rows = enrich_manifest(load_jsonl(manifest))
    scores = load_scores(score_path)
    attach_scores(rows, scores)

    df = pd.DataFrame(rows)

    unparsed = int(df["baseline_correct"].isna().sum())
    if unparsed:
        raise RuntimeError(
            f"{unparsed}/{len(df)} baseline answers could not be parsed from the 64-step manifest."
        )

    base_acc = float(df["baseline_correct"].astype(float).mean())
    curve = selective_curve(df)

    df.to_csv(out / "csqa64_per_sample.csv", index=False)
    curve.to_csv(out / "csqa64_selective_curve.csv", index=False)

    q25 = curve[np.isclose(curve["abstain_rate"], 0.25)].iloc[0]

    summary = {
        "n": int(len(df)),
        "manifest": str(manifest),
        "rcc_scores": str(score_path),
        "baseline_accuracy": base_acc,
        "top25_abstention": {
            "coverage": float(q25["coverage"]),
            "answered_accuracy": float(q25["answered_accuracy"]),
            "answered_error_rate": float(q25["answered_error_rate"]),
            "relative_error_reduction": float(q25["relative_error_reduction"]),
            "abstained_subset_accuracy": float(q25["abstained_subset_accuracy"]),
        },
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 64-step curve
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        curve["coverage"] * 100,
        curve["answered_accuracy"] * 100,
        marker="o",
        label="LLaDA-64",
    )
    ax.axhline(
        base_acc * 100,
        linestyle="--",
        linewidth=1,
        label=f"64-step baseline {100*base_acc:.1f}%",
    )
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy on answered samples (%)")
    ax.set_title("CSQA: RCC selective answering at 64 denoising steps")
    ax.grid(True, alpha=0.25)
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "csqa64_selective_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Optional 64/128 comparison.
    ref128 = Path(args.reference128)
    if ref128.exists():
        d128 = pd.read_csv(ref128)
        if {"rcc_score", "baseline_correct"}.issubset(d128.columns):
            d128 = d128[d128["baseline_correct"].notna()].copy()
            c128 = selective_curve(d128)
            c128.to_csv(out / "csqa128_selective_curve_recomputed.csv", index=False)

            b128 = float(d128["baseline_correct"].astype(float).mean())
            q128 = c128[np.isclose(c128["abstain_rate"], 0.25)].iloc[0]

            comp = pd.DataFrame([
                {
                    "setting": "LLaDA-64",
                    "baseline_accuracy": base_acc,
                    "coverage_at_25pct_abstention": float(q25["coverage"]),
                    "selective_accuracy": float(q25["answered_accuracy"]),
                    "relative_error_reduction": float(q25["relative_error_reduction"]),
                    "abstained_subset_accuracy": float(q25["abstained_subset_accuracy"]),
                },
                {
                    "setting": "LLaDA-128",
                    "baseline_accuracy": b128,
                    "coverage_at_25pct_abstention": float(q128["coverage"]),
                    "selective_accuracy": float(q128["answered_accuracy"]),
                    "relative_error_reduction": float(q128["relative_error_reduction"]),
                    "abstained_subset_accuracy": float(q128["abstained_subset_accuracy"]),
                },
            ])
            comp.to_csv(out / "csqa64_vs_128.csv", index=False)

            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            ax.plot(
                curve["coverage"] * 100,
                curve["answered_accuracy"] * 100,
                marker="o",
                label="LLaDA-64",
            )
            ax.plot(
                c128["coverage"] * 100,
                c128["answered_accuracy"] * 100,
                marker="o",
                label="LLaDA-128",
            )
            ax.set_xlabel("Coverage (%)")
            ax.set_ylabel("Accuracy on answered samples (%)")
            ax.set_title("CSQA: RCC mitigation across denoising horizons")
            ax.grid(True, alpha=0.25)
            ax.invert_xaxis()
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "csqa64_vs_128_curve.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    print("=" * 108)
    print("RESULT: CSQA-64")
    print("=" * 108)
    print(f"n                         : {len(df)}")
    print(f"baseline accuracy         : {100*base_acc:.2f}%")
    print("reject top 25% RCC risk:")
    print(f"  coverage                : {100*float(q25['coverage']):.2f}%")
    print(f"  answered accuracy       : {100*float(q25['answered_accuracy']):.2f}%")
    print(
        f"  relative error reduction: "
        f"{100*float(q25['relative_error_reduction']):.2f}%"
    )
    print(
        f"  rejected subset accuracy: "
        f"{100*float(q25['abstained_subset_accuracy']):.2f}%"
    )
    print()
    print(f"Saved: {out / 'summary.json'}")
    print(f"Saved: {out / 'csqa64_selective_curve.csv'}")
    print(f"Saved: {out / 'csqa64_selective_curve.png'}")
    if (out / "csqa64_vs_128.csv").exists():
        print(f"Saved: {out / 'csqa64_vs_128.csv'}")
        print(f"Saved: {out / 'csqa64_vs_128_curve.png'}")


if __name__ == "__main__":
    main()
