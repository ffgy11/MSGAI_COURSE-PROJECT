#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from test_pcs_selection_compact import (
    evaluate_dataset,
    infer_dataset_from_manifest,
    load_manifest,
    parse_label,
    resolve_entropy_file,
)

TRACEDET_LLADA128 = {
    "triviaqa": 0.739,
    "hotpotqa": 0.661,
    "commonsenseqa": 0.772,
}

def infer_model_family(path: Path) -> str:
    s = str(path).lower()
    if "dream" in s:
        return "dream"
    if "llada" in s or "exp2_" in s or "exp_" in s:
        return "llada"
    return "unknown"

def infer_steps_from_path(path: Path) -> Optional[int]:
    s = str(path).lower()
    for pat in [r"steps[_-]?(64|128|256)", r"step[_-]?(64|128|256)", r"[_/-](64|128|256)[_/.-]"]:
        m = re.search(pat, s)
        if m:
            return int(m.group(1))
    return None

def infer_steps_from_cache(manifest: Path, repo_root: Optional[Path]) -> Optional[int]:
    try:
        rows = load_manifest(manifest)
    except Exception:
        return None
    for row in rows[:20]:
        if parse_label(row) is None:
            continue
        try:
            p = resolve_entropy_file(row, manifest, repo_root)
            with np.load(p, allow_pickle=False) as z:
                if "entropy" in z.files:
                    a = np.asarray(z["entropy"])
                    if a.ndim >= 1:
                        return int(a.shape[0])
        except Exception:
            continue
    return None

def count_labeled_rows(path: Path) -> int:
    try:
        rows = load_manifest(path)
    except Exception:
        return 0
    return sum(parse_label(r) is not None for r in rows)

def candidate_score(path: Path, model: str, steps: int, dataset: str, n_labeled: int) -> float:
    s = str(path).lower()
    score = min(n_labeled, 2000) / 100.0
    if model == "dream":
        if "dream" in s:
            score += 30
        if "official" in s:
            score += 5
        if "1000" in s:
            score += 6
    if model == "llada":
        if "exp2" in s:
            score += 8
        if "400" in s:
            score += 5
    if f"steps{steps}" in s:
        score += 8
    elif str(steps) in s:
        score += 2
    if dataset == "triviaqa" and ("tqa" in s or "trivia" in s):
        score += 4
    if dataset == "hotpotqa" and "hotpot" in s:
        score += 4
    if dataset == "commonsenseqa" and ("csqa" in s or "commonsense" in s):
        score += 4
    if "pcs_" in s or "rcc_" in s or "cci_" in s:
        score -= 100
    return score

def discover_configs(root: Path, repo_root: Optional[Path]) -> Tuple[Dict[Tuple[str,int,str],Path], List[dict]]:
    grouped = defaultdict(list)
    diagnostics = []
    for p in root.rglob("manifest.jsonl"):
        ds = infer_dataset_from_manifest(p)
        if ds not in {"triviaqa","hotpotqa","commonsenseqa"}:
            continue
        model = infer_model_family(p)
        steps = infer_steps_from_path(p)
        if steps is None:
            steps = infer_steps_from_cache(p, repo_root)
        if steps is None:
            continue
        if model == "unknown":
            s = str(p).lower()
            if "exp" in s:
                model = "llada"
        if model not in {"llada","dream"}:
            continue
        n = count_labeled_rows(p)
        sc = candidate_score(p, model, steps, ds, n)
        grouped[(model, int(steps), ds)].append((sc, n, p.resolve()))
        diagnostics.append({
            "model": model, "steps": int(steps), "dataset": ds,
            "n_labeled": int(n), "score": float(sc),
            "manifest": str(p.resolve())
        })
    selected = {}
    for key, arr in grouped.items():
        arr.sort(key=lambda x: (x[0], x[1], str(x[2])), reverse=True)
        selected[key] = arr[0][2]
    return selected, diagnostics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto_root", default="outputs")
    ap.add_argument("--repo_root", default=None)
    ap.add_argument("--output_dir", default="outputs/rcc_all_configs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    auto_root = Path(args.auto_root)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    configs, diagnostics = discover_configs(auto_root, repo_root)
    (out_root / "manifest_candidates.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    if not configs:
        raise RuntimeError(f"No LLaDA/Dream manifests discovered under {auto_root}")

    print("="*120)
    print("RCC ALL-CONFIG EVALUATION")
    print("Frozen 16 features, same 5-fold OOF LogisticRegression everywhere")
    print("="*120)
    print("\nSELECTED MANIFESTS")
    for (model, steps, ds), p in sorted(configs.items()):
        print(f"  {model:6s} steps={steps:<3d} {ds:16s} -> {p}")

    all_rows = []
    for (model, steps, ds), manifest in sorted(configs.items()):
        config_name = f"{model}_steps{steps}"
        config_out = out_root / config_name

        print("\n" + "-"*120)
        print(f"Evaluating model={model} steps={steps} dataset={ds}")

        rows = evaluate_dataset(
            dataset=ds,
            manifest_path=manifest,
            repo_root=repo_root,
            out_root=config_out,
            folds=args.folds,
            seed=args.seed,
            bootstrap=args.bootstrap,
        )

        tracedet_ref = TRACEDET_LLADA128.get(ds) if (model == "llada" and steps == 128) else None

        for r0 in rows:
            r = dict(r0)
            r["model"] = model
            r["steps"] = int(steps)
            r["config"] = config_name
            r["tracedet_reference_auroc"] = tracedet_ref
            r["delta_vs_tracedet"] = np.nan if tracedet_ref is None else float(r["auroc"]) - tracedet_ref
            all_rows.append(r)
            print(
                f"{r['method']:22s} "
                f"AUROC={100*r['auroc']:6.2f} "
                f"AUPRC={100*r['auprc']:6.2f} "
                f"FPR95={100*r['fpr95']:6.2f}"
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_root / "all_results.csv", index=False)

    print("\n" + "="*120)
    print("MACRO AUROC BY CONFIG")
    print("="*120)

    macro_rows = []
    for (model, steps), sub in df.groupby(["model","steps"]):
        print(f"\n{model.upper()} - {steps} steps")
        for method in ["pcs_only","selection_only","pcs_plus_selection","tre_reference"]:
            m = sub[sub["method"] == method]
            if len(m) == 0:
                continue
            macro_auc = float(m["auroc"].mean())
            macro_ap = float(m["auprc"].mean())
            print(f"  {method:22s} macro AUROC={100*macro_auc:6.2f} macro AUPRC={100*macro_ap:6.2f}")
            macro_rows.append({
                "model": model, "steps": int(steps), "method": method,
                "macro_auroc": macro_auc, "macro_auprc": macro_ap,
                "n_datasets": int(len(m))
            })

    pd.DataFrame(macro_rows).to_csv(out_root / "macro_summary.csv", index=False)

    print("\n" + "="*120)
    print("RCC (PCS+SELECTION) PER DATASET")
    print("="*120)

    rcc = df[df["method"] == "pcs_plus_selection"].copy()
    rcc = rcc.sort_values(["model","steps","dataset"])
    if len(rcc):
        show = rcc[["model","steps","dataset","n","auroc","auprc","fpr95"]].copy()
        for c in ["auroc","auprc","fpr95"]:
            show[c] = 100*show[c]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nSaved: {out_root / 'all_results.csv'}")
    print(f"Saved: {out_root / 'macro_summary.csv'}")
    print(f"Saved: {out_root / 'manifest_candidates.json'}")

if __name__ == "__main__":
    main()
