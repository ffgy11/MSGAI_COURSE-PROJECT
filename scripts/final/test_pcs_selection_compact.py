#!/usr/bin/env python3
"""
PCS + Selection Consistency (compact, frozen feature set)

Purpose
-------
Course-project detector for dLLM hallucination detection that is distinct from TRE.

Three fixed detectors are evaluated:
  1) PCS-only: 8 pre-commitment stability features
  2) Selection-only: 8 reveal-choice consistency features
  3) PCS+Selection: all 16 fixed features

No TRE feature is used in any detector.
TRE is recomputed only as an external reference.

PCS features (window=8):
  - late_jump_mean
  - late_jump_q90
  - all_jump_mean
  - all_jump_std
  - all_jump_max
  - all_volatility_max
  - all_volatility_std
  - all_peak_overshoot_max

Selection features:
  - sel_percentile_mean
  - sel_percentile_late
  - sel_gap_mean
  - sel_gap_late
  - sel_overlap_mean
  - sel_overlap_late
  - sel_regret_norm_mean
  - sel_regret_norm_late

Definitions at denoising step t
-------------------------------
Let U_t be unresolved generation positions before reveal.
Let R_t be positions selected to reveal at step t, k=|R_t|.

1) selection percentile:
   percentile rank of each selected token's entropy within U_t,
   averaged over selected tokens.
   Lower = decoder selected relatively confident tokens.

2) selected-vs-remaining gap:
   mean entropy(selected) - mean entropy(unselected unresolved)
   Lower/more negative = selected tokens are more confident.

3) optimal overlap:
   |R_t ∩ K_t| / k
   where K_t are the k lowest-entropy tokens in U_t.
   Higher = reveal decision agrees with greedy confidence selection.

4) normalized regret:
   (mean entropy(selected) - mean entropy(K_t)) / (std entropy(U_t)+eps)
   Lower = reveal choice is closer to entropy-optimal.

Aggregation:
  mean over all valid steps, and mean over late third of steps.
  PCS additionally uses fixed response-level aggregations above.

Classifier:
  Fixed 5-fold stratified OOF LogisticRegression, C=1.0.
  Same feature set and hyperparameters across all datasets.

Usage:
  python scripts/test_pcs_selection_compact.py \
    --auto_root outputs \
    --repo_root "$(pwd)" \
    --output_dir outputs/pcs_selection_compact
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
PCS_WINDOW = 8

PCS_FEATURES = [
    "late_jump_mean",
    "late_jump_q90",
    "all_jump_mean",
    "all_jump_std",
    "all_jump_max",
    "all_volatility_max",
    "all_volatility_std",
    "all_peak_overshoot_max",
]

SEL_FEATURES = [
    "sel_percentile_mean",
    "sel_percentile_late",
    "sel_gap_mean",
    "sel_gap_late",
    "sel_overlap_mean",
    "sel_overlap_late",
    "sel_regret_norm_mean",
    "sel_regret_norm_late",
]

ALL_FEATURES = PCS_FEATURES + SEL_FEATURES

TRACEDET_REFERENCE = {
    "triviaqa": 0.739,
    "hotpotqa": 0.661,
    "commonsenseqa": 0.772,
}


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def canonical_dataset_name(x: str) -> str:
    s = str(x).strip().lower().replace("-", "").replace("_", "")
    if s in {"triviaqa", "trivia", "tqa"}:
        return "triviaqa"
    if s in {"hotpotqa", "hotpot", "hpqa"}:
        return "hotpotqa"
    if s in {"commonsenseqa", "commonsense", "csqa"}:
        return "commonsenseqa"
    return str(x).strip().lower()


def parse_label(row: dict) -> Optional[int]:
    h = row.get("hallucination", None)

    if isinstance(h, dict):
        consistent = h.get("consistent", None)
        if consistent is True:
            return 0
        if consistent is False:
            return 1

        s = str(h.get("label", "")).strip().lower()
        if s in {"hallucination", "hallucinated", "hall", "1", "true"}:
            return 1
        if s in {
            "non_hallucination", "non-hallucination", "nonhallucination",
            "factual", "correct", "0", "false",
        }:
            return 0

    if isinstance(h, bool):
        return int(h)

    if isinstance(h, (int, np.integer)):
        x = int(h)
        return x if x in (0, 1) else None

    for key in ("label", "is_hallucination", "hallucinated"):
        if key not in row:
            continue
        v = row[key]
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, np.integer)) and int(v) in (0, 1):
            return int(v)

    return None


def load_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_entropy_file(
    row: dict,
    manifest_path: Path,
    repo_root: Optional[Path],
) -> Path:
    raw = row.get("entropy_file")
    if raw is None:
        raise KeyError("manifest row missing entropy_file")

    p = Path(raw)
    if p.is_file():
        return p.resolve()

    candidates = [
        manifest_path.parent / p,
        manifest_path.parent / p.name,
        manifest_path.parent / "entropy" / p.name,
        manifest_path.parent / "npz" / p.name,
    ]

    if repo_root is not None:
        parts = list(p.parts)
        if "outputs" in parts:
            k = parts.index("outputs")
            candidates.append(repo_root / Path(*parts[k:]))

    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Cannot resolve entropy_file={raw!r} from manifest={manifest_path}"
    )


def scalar_npz(z, key: str, default: int) -> int:
    if key not in z.files:
        return int(default)
    a = np.asarray(z[key]).reshape(-1)
    return int(a[0]) if len(a) else int(default)


# -----------------------------------------------------------------------------
# TRE reference only
# -----------------------------------------------------------------------------

def official_tre(z) -> float:
    entropy = np.asarray(z["entropy"], dtype=np.float64)
    counts = np.asarray(z["revealed_counts"], dtype=np.int64).reshape(-1)
    pos = np.asarray(z["revealed_positions"], dtype=np.int64)

    T, L = entropy.shape
    mass = np.zeros(T, dtype=np.float64)

    for t in range(T):
        n = int(counts[t])
        if n <= 0:
            continue

        p = np.asarray(pos[t, :n], dtype=np.int64)
        p = p[(p >= 0) & (p < L)]
        if len(p) == 0:
            continue

        p = np.unique(p)
        vals = entropy[t, p]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            mass[t] = float(vals.sum())

    w = (np.arange(T, dtype=np.float64) + 1.0) / float(T)
    return float(np.dot(w, mass))


# -----------------------------------------------------------------------------
# Reveal reconstruction
# -----------------------------------------------------------------------------

def reveal_times(
    counts: np.ndarray,
    positions_abs: np.ndarray,
    prompt_len: int,
    gen_length: int,
) -> np.ndarray:
    rt = np.full(gen_length, -1, dtype=np.int64)

    for t in range(len(counts)):
        n = int(counts[t])
        if n <= 0:
            continue

        p = np.asarray(positions_abs[t, :n], dtype=np.int64) - prompt_len
        p = p[(p >= 0) & (p < gen_length)]

        for j in np.unique(p):
            if rt[j] < 0:
                rt[j] = t

    return rt


# -----------------------------------------------------------------------------
# PCS features
# -----------------------------------------------------------------------------

def token_pcs_stats(
    H: np.ndarray,
    token_idx: int,
    reveal_step: int,
    window: int = PCS_WINDOW,
) -> Tuple[float, float, float]:
    if reveal_step <= 0:
        return 0.0, 0.0, 0.0

    s = max(0, reveal_step - window)
    x = np.asarray(H[s:reveal_step + 1, token_idx], dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)

    if len(x) < 2:
        return 0.0, 0.0, 0.0

    # Positive => final-step entropy drop => confidence jump.
    last_jump = float(x[-2] - x[-1])

    # Local instability before commitment.
    volatility = float(np.mean(np.abs(np.diff(x))))

    # Earlier confidence gain that was later corrected before final commitment.
    gain = x[0] - x
    peak_overshoot = float(np.max(gain) - gain[-1])

    return last_jump, volatility, peak_overshoot


def compute_pcs_features(
    H: np.ndarray,
    reveal_t: np.ndarray,
) -> Dict[str, float]:
    T, N = H.shape
    valid = np.where(reveal_t >= 0)[0]
    if len(valid) == 0:
        raise RuntimeError("no reveal tokens for PCS")

    jumps = []
    vols = []
    overs = []
    fracs = []

    for i in valid:
        r = int(reveal_t[i])
        j, v, o = token_pcs_stats(H, int(i), r, PCS_WINDOW)
        jumps.append(j)
        vols.append(v)
        overs.append(o)
        fracs.append(r / max(T - 1, 1))

    jumps = np.asarray(jumps, dtype=np.float64)
    vols = np.asarray(vols, dtype=np.float64)
    overs = np.asarray(overs, dtype=np.float64)
    fracs = np.asarray(fracs, dtype=np.float64)

    late = fracs >= (2.0 / 3.0)
    late_jumps = jumps[late] if np.any(late) else jumps

    return {
        "late_jump_mean": float(np.mean(late_jumps)),
        "late_jump_q90": float(np.quantile(late_jumps, 0.90)),
        "all_jump_mean": float(np.mean(jumps)),
        "all_jump_std": float(np.std(jumps)),
        "all_jump_max": float(np.max(jumps)),
        "all_volatility_max": float(np.max(vols)),
        "all_volatility_std": float(np.std(vols)),
        "all_peak_overshoot_max": float(np.max(overs)),
    }


# -----------------------------------------------------------------------------
# Selection Consistency features
# -----------------------------------------------------------------------------

def average_percentile_ranks(values: np.ndarray, selected_local: np.ndarray) -> float:
    """
    values correspond to all currently unresolved tokens.
    selected_local are indices into values.

    Returns average percentile rank in [0,1].
    0 means among the lowest-entropy / most confident tokens.
    """
    m = len(values)
    if m <= 1 or len(selected_local) == 0:
        return 0.0

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(m, dtype=np.float64)
    ranks[order] = np.arange(m, dtype=np.float64)

    # tie handling: replace ranks by average tie rank
    uniq, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for uidx, c in enumerate(counts):
            if c <= 1:
                continue
            idx = np.where(inv == uidx)[0]
            ranks[idx] = np.mean(ranks[idx])

    pct = ranks / max(m - 1, 1)
    return float(np.mean(pct[selected_local]))


def compute_selection_features(
    H: np.ndarray,
    counts: np.ndarray,
    positions_abs: np.ndarray,
    prompt_len: int,
    reveal_t: np.ndarray,
) -> Tuple[Dict[str, float], dict]:
    """
    Uses the reconstructed reveal schedule to define U_t exactly:
      unresolved before step t = tokens with reveal_time >= t (or never revealed).

    Selection event R_t comes from revealed_positions.
    """
    T, N = H.shape

    per_step = []
    valid_steps = 0

    for t in range(T):
        n = int(counts[t])
        if n <= 0:
            continue

        sel = np.asarray(positions_abs[t, :n], dtype=np.int64) - prompt_len
        sel = np.unique(sel[(sel >= 0) & (sel < N)])
        if len(sel) == 0:
            continue

        unresolved = np.where((reveal_t < 0) | (reveal_t >= t))[0]
        if len(unresolved) == 0:
            continue

        # Ensure selected are included in unresolved candidate set.
        sel = np.intersect1d(sel, unresolved, assume_unique=False)
        if len(sel) == 0:
            continue

        vals = np.asarray(H[t, unresolved], dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, 0.0)

        # map generation-relative token index -> local index in unresolved array
        local_map = {int(tok): j for j, tok in enumerate(unresolved)}
        sel_local = np.asarray([local_map[int(tok)] for tok in sel], dtype=np.int64)

        k = len(sel_local)

        # 1) percentile
        sel_percentile = average_percentile_ranks(vals, sel_local)

        # 2) selected-vs-remaining entropy gap
        selected_vals = vals[sel_local]
        selected_mask = np.zeros(len(unresolved), dtype=bool)
        selected_mask[sel_local] = True
        remaining_vals = vals[~selected_mask]

        if len(remaining_vals):
            sel_gap = float(np.mean(selected_vals) - np.mean(remaining_vals))
        else:
            sel_gap = 0.0

        # 3) optimal overlap with k lowest-entropy candidates
        k_eff = min(k, len(vals))
        opt_local = np.argpartition(vals, k_eff - 1)[:k_eff]
        overlap = len(np.intersect1d(sel_local, opt_local))
        sel_overlap = float(overlap / max(k_eff, 1))

        # 4) normalized regret relative to entropy-greedy selection
        optimal_mean = float(np.mean(vals[opt_local]))
        selected_mean = float(np.mean(selected_vals))
        scale = float(np.std(vals))
        sel_regret_norm = float((selected_mean - optimal_mean) / max(scale, EPS))

        per_step.append({
            "tfrac": t / max(T - 1, 1),
            "percentile": sel_percentile,
            "gap": sel_gap,
            "overlap": sel_overlap,
            "regret_norm": sel_regret_norm,
        })
        valid_steps += 1

    if not per_step:
        raise RuntimeError("no valid reveal-selection steps")

    def agg(name: str, late_only: bool) -> float:
        xs = [
            d[name]
            for d in per_step
            if (not late_only) or d["tfrac"] >= (2.0 / 3.0)
        ]
        if not xs:
            xs = [d[name] for d in per_step]
        return float(np.mean(xs))

    feat = {
        "sel_percentile_mean": agg("percentile", False),
        "sel_percentile_late": agg("percentile", True),
        "sel_gap_mean": agg("gap", False),
        "sel_gap_late": agg("gap", True),
        "sel_overlap_mean": agg("overlap", False),
        "sel_overlap_late": agg("overlap", True),
        "sel_regret_norm_mean": agg("regret_norm", False),
        "sel_regret_norm_late": agg("regret_norm", True),
    }

    return feat, {
        "selection_valid_steps": int(valid_steps),
    }


# -----------------------------------------------------------------------------
# Per-sample extraction
# -----------------------------------------------------------------------------

def extract_features(npz_path: Path) -> Tuple[Dict[str, float], float, dict]:
    with np.load(npz_path, allow_pickle=False) as z:
        required = {"entropy", "revealed_counts", "revealed_positions"}
        missing = required - set(z.files)
        if missing:
            raise KeyError(f"{npz_path}: missing {sorted(missing)}")

        entropy = np.asarray(z["entropy"], dtype=np.float64)
        counts = np.asarray(z["revealed_counts"], dtype=np.int64).reshape(-1)
        positions = np.asarray(z["revealed_positions"], dtype=np.int64)

        T, L = entropy.shape
        prompt_len = scalar_npz(z, "prompt_len", 0)
        gen_length = scalar_npz(z, "gen_length", L - prompt_len)

        start = max(0, prompt_len)
        N = min(gen_length, L - start)
        if N <= 0:
            raise ValueError("invalid generation slice")

        H = entropy[:, start:start + N].copy()
        H[~np.isfinite(H)] = 0.0
        H = np.maximum(H, 0.0)

        rt = reveal_times(counts, positions, prompt_len, N)

        pcs = compute_pcs_features(H, rt)
        sel, sel_audit = compute_selection_features(
            H=H,
            counts=counts,
            positions_abs=positions,
            prompt_len=prompt_len,
            reveal_t=rt,
        )

        feat = {**pcs, **sel}
        tre = official_tre(z)

        audit = {
            "T": int(T),
            "N": int(N),
            "revealed_tokens": int(np.sum(rt >= 0)),
            "missing_reveal_tokens": int(np.sum(rt < 0)),
            **sel_audit,
        }

        return feat, tre, audit


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def build_model(seed: int):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )),
    ])


def oof_predict(
    X: np.ndarray,
    y: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    pred = np.full(len(y), np.nan, dtype=np.float64)

    for fold, (tr, te) in enumerate(cv.split(X, y)):
        model = build_model(seed + 1000 * fold)
        model.fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]

    if np.isnan(pred).any():
        raise RuntimeError("incomplete OOF predictions")

    return pred


def metric_dict(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(y, score)
    ok = np.where(tpr >= 0.95)[0]

    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "fpr95": float(fpr[ok[0]]) if len(ok) else 1.0,
    }


def paired_bootstrap(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue

        vals.append(
            roc_auc_score(yy, a[idx])
            - roc_auc_score(yy, b[idx])
        )

    vals = np.asarray(vals, dtype=np.float64)

    return {
        "delta_mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "p_gt_0": float(np.mean(vals > 0)),
    }


# -----------------------------------------------------------------------------
# Manifest discovery
# -----------------------------------------------------------------------------

def infer_dataset_from_manifest(path: Path) -> Optional[str]:
    try:
        rows = load_manifest(path)
    except Exception:
        return None

    if not rows:
        return None

    for key in ("dataset", "dataset_name", "benchmark"):
        if key in rows[0] and rows[0][key]:
            ds = canonical_dataset_name(rows[0][key])
            if ds in TRACEDET_REFERENCE:
                return ds

    s = str(path).lower()

    if "trivia" in s or re.search(r"(^|[_/])tqa([_/]|$)", s):
        return "triviaqa"
    if "hotpot" in s:
        return "hotpotqa"
    if "commonsense" in s or "csqa" in s:
        return "commonsenseqa"

    return None


def score_manifest_candidate(path: Path, ds: str) -> int:
    s = str(path).lower()
    score = 0

    if "128" in s:
        score += 5
    if "steps128" in s:
        score += 4
    if "400" in s:
        score += 4
    if "exp2" in s:
        score += 2

    if ds == "triviaqa" and ("trivia" in s or "tqa" in s):
        score += 5
    if ds == "hotpotqa" and "hotpot" in s:
        score += 5
    if ds == "commonsenseqa" and ("commonsense" in s or "csqa" in s):
        score += 5

    return score


def auto_discover_manifests(root: Path) -> Dict[str, Path]:
    candidates: Dict[str, List[Tuple[int, Path]]] = {
        "triviaqa": [],
        "hotpotqa": [],
        "commonsenseqa": [],
    }

    for p in root.rglob("manifest.jsonl"):
        ds = infer_dataset_from_manifest(p)
        if ds in candidates:
            candidates[ds].append(
                (score_manifest_candidate(p, ds), p)
            )

    chosen = {}

    for ds, arr in candidates.items():
        if not arr:
            continue
        arr.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
        chosen[ds] = arr[0][1].resolve()

    return chosen


def parse_explicit_manifests(items: List[str]) -> Dict[str, Path]:
    out = {}

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"--manifest must be DATASET=PATH, got {item!r}"
            )

        ds, raw = item.split("=", 1)
        ds = canonical_dataset_name(ds)

        if ds not in TRACEDET_REFERENCE:
            raise ValueError(f"unsupported dataset: {ds}")

        out[ds] = Path(raw).resolve()

    return out


# -----------------------------------------------------------------------------
# Dataset evaluation
# -----------------------------------------------------------------------------

def evaluate_dataset(
    dataset: str,
    manifest_path: Path,
    repo_root: Optional[Path],
    out_root: Path,
    folds: int,
    seed: int,
    bootstrap: int,
) -> List[dict]:
    rows = load_manifest(manifest_path)

    feats = []
    labels = []
    tres = []
    metas = []
    audits = []
    failures = []

    for row in rows:
        y = parse_label(row)
        if y is None:
            continue

        try:
            p = resolve_entropy_file(
                row,
                manifest_path,
                repo_root,
            )
            f, tre, audit = extract_features(p)

            feats.append(f)
            labels.append(int(y))
            tres.append(float(tre))
            audits.append(audit)

            metas.append({
                "index": row.get("index"),
                "question_id": row.get(
                    "question_id",
                    row.get("index"),
                ),
                "entropy_file": str(p),
            })

        except Exception as e:
            failures.append({
                "index": row.get("index"),
                "error": repr(e),
            })

    if not feats:
        raise RuntimeError(f"{dataset}: no valid samples")

    y = np.asarray(labels, dtype=np.int64)
    tre = np.asarray(tres, dtype=np.float64)

    X_all = np.asarray(
        [[f[name] for name in ALL_FEATURES] for f in feats],
        dtype=np.float64,
    )

    X_pcs = X_all[:, :len(PCS_FEATURES)]
    X_sel = X_all[:, len(PCS_FEATURES):]

    pcs = oof_predict(X_pcs, y, folds, seed)
    sel = oof_predict(X_sel, y, folds, seed)
    combo = oof_predict(X_all, y, folds, seed)

    scores = {
        "pcs_only": pcs,
        "selection_only": sel,
        "pcs_plus_selection": combo,
        "tre_reference": tre,
    }

    ref = TRACEDET_REFERENCE.get(dataset)
    result_rows = []

    for method, score in scores.items():
        m = metric_dict(y, score)

        row = {
            "dataset": dataset,
            "method": method,
            "n": int(len(y)),
            "hallucination": int(y.sum()),
            "non_hallucination": int((1-y).sum()),
            "auroc": m["auroc"],
            "auprc": m["auprc"],
            "fpr95": m["fpr95"],
            "tracedet_reference_auroc": ref,
            "delta_vs_tracedet": (
                float(m["auroc"] - ref)
                if ref is not None else np.nan
            ),
            "failures": len(failures),
        }

        if method != "tre_reference":
            b = paired_bootstrap(
                y=y,
                a=score,
                b=tre,
                n_boot=bootstrap,
                seed=seed + 777,
            )
            row.update({
                "delta_vs_tre_boot_mean": b["delta_mean"],
                "delta_vs_tre_ci_low": b["ci_low"],
                "delta_vs_tre_ci_high": b["ci_high"],
                "p_delta_vs_tre_gt_0": b["p_gt_0"],
            })

        result_rows.append(row)

    ds_out = out_root / dataset
    ds_out.mkdir(parents=True, exist_ok=True)

    sample_df = pd.DataFrame(metas)
    sample_df["label"] = y

    for j, name in enumerate(ALL_FEATURES):
        sample_df[name] = X_all[:, j]

    for method, score in scores.items():
        sample_df[f"score__{method}"] = score

    sample_df.to_csv(
        ds_out / "sample_scores.csv",
        index=False,
    )

    with (ds_out / "audit.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({
            "dataset": dataset,
            "manifest": str(manifest_path),
            "n": int(len(y)),
            "n_hallucination": int(y.sum()),
            "n_non_hallucination": int((1-y).sum()),
            "pcs_features": PCS_FEATURES,
            "selection_features": SEL_FEATURES,
            "pcs_window": PCS_WINDOW,
            "folds": folds,
            "seed": seed,
            "failures": failures,
            "sample_audits": audits,
        }, f, indent=2)

    return result_rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="DATASET=PATH; may repeat.",
    )
    ap.add_argument("--auto_root", default=None)
    ap.add_argument("--repo_root", default=None)
    ap.add_argument(
        "--output_dir",
        default="outputs/pcs_selection_compact",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)

    args = ap.parse_args()

    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root else None
    )
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifests = parse_explicit_manifests(args.manifest)

    if args.auto_root:
        auto = auto_discover_manifests(Path(args.auto_root))
        for ds, p in auto.items():
            manifests.setdefault(ds, p)

    if not manifests:
        raise RuntimeError("No manifests found.")

    print("=" * 112)
    print("PCS + SELECTION CONSISTENCY: COMPACT FROZEN DETECTOR")
    print("=" * 112)

    print("PCS FEATURES")
    for i, name in enumerate(PCS_FEATURES, 1):
        print(f"  {i:2d}. {name}")

    print()
    print("SELECTION FEATURES")
    for i, name in enumerate(SEL_FEATURES, 1):
        print(f"  {i:2d}. {name}")

    print()
    print("MANIFESTS")
    for ds in ("triviaqa", "hotpotqa", "commonsenseqa"):
        print(
            f"  {ds:16s}: "
            f"{manifests.get(ds, 'NOT FOUND')}"
        )

    all_results = []

    for ds in ("triviaqa", "hotpotqa", "commonsenseqa"):
        if ds not in manifests:
            continue

        print()
        print("-" * 112)
        print(f"Evaluating {ds}")

        rows = evaluate_dataset(
            dataset=ds,
            manifest_path=manifests[ds],
            repo_root=repo_root,
            out_root=out_root,
            folds=args.folds,
            seed=args.seed,
            bootstrap=args.bootstrap,
        )

        all_results.extend(rows)

        for r in rows:
            print(
                f"{r['method']:22s} "
                f"AUROC={100*r['auroc']:6.2f} "
                f"AUPRC={100*r['auprc']:6.2f} "
                f"FPR95={100*r['fpr95']:6.2f} "
                f"ΔTraceDet={100*r['delta_vs_tracedet']:+6.2f}"
            )

    df = pd.DataFrame(all_results)
    df.to_csv(out_root / "summary.csv", index=False)

    print()
    print("=" * 112)
    print("MACRO AUROC")
    print("=" * 112)

    methods = [
        "pcs_only",
        "selection_only",
        "pcs_plus_selection",
        "tre_reference",
    ]

    for method in methods:
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue

        macro = float(sub["auroc"].mean())
        ref = float(
            sub["tracedet_reference_auroc"].mean()
        )

        print(
            f"{method:22s} "
            f"macro AUROC={100*macro:6.2f} | "
            f"TraceDet-ref={100*ref:6.2f} | "
            f"delta={100*(macro-ref):+6.2f}"
        )

    print()
    print(f"Saved: {out_root / 'summary.csv'}")
    print()
    print("DECISION RULE")
    print("  If PCS+Selection improves macro AUROC and fixes/helps CSQA, use it.")
    print("  If not, keep PCS-only and stop feature engineering.")


if __name__ == "__main__":
    main()
