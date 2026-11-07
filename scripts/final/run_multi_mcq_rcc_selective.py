#!/usr/bin/env python3
"""
Multi-dataset RCC selective mitigation sweep for LLaDA.

Datasets (all 4-choice):
  - ARC-Easy
  - OpenBookQA
  - SciQ
  - HellaSwag

For each dataset:
  1) sample N validation examples with deterministic seed;
  2) run LLaDA diffusion decoding and collect the fixed 16-D RCC statistics;
  3) label each generation by exact multiple-choice ground truth (no LLM judge);
  4) split 50/50 train/test, fit a lightweight logistic-regression RCC detector;
  5) evaluate selective answering on held-out test;
  6) report risk-coverage / accuracy and relative error reduction.

The script is resumable. Each completed generation is appended to JSONL and reused.
No base-model fine-tuning is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import matplotlib.pyplot as plt


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MASK_ID_DEFAULT = 126336

FEATURE_NAMES = [
    "late_jump_mean",
    "late_jump_q90",
    "all_jump_mean",
    "all_jump_std",
    "all_jump_max",
    "all_volatility_max",
    "all_volatility_std",
    "all_peak_overshoot_max",
    "sel_percentile_mean",
    "sel_percentile_late",
    "sel_gap_mean",
    "sel_gap_late",
    "sel_overlap_mean",
    "sel_overlap_late",
    "sel_regret_norm_mean",
    "sel_regret_norm_late",
]

DATASET_CHOICES = ["arc_easy", "openbookqa", "sciq", "hellaswag"]


def dump_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_answer_key(answer_key: Any, original_labels: list[str], n: int) -> int:
    """Return zero-based gold index robustly."""
    if isinstance(answer_key, (int, np.integer)):
        z = int(answer_key)
        if 0 <= z < n:
            return z

    s = str(answer_key).strip()
    # Exact source label first (ARC occasionally uses numeric labels).
    if s in original_labels:
        return original_labels.index(s)
    su = s.upper()
    if su in original_labels:
        return original_labels.index(su)
    # Standard A/B/C/D answer key.
    if len(su) == 1 and su in LETTERS[:n]:
        return LETTERS.index(su)
    # 0/1/2/3 or 1/2/3/4 numeric fallback.
    if s.isdigit():
        z = int(s)
        if 0 <= z < n:
            return z
        if 1 <= z <= n:
            return z - 1
    raise ValueError(f"Cannot map answer key {answer_key!r} with labels={original_labels}")


def load_mcq_rows(name: str, max_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    if name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        raw = []
        for i, r in enumerate(ds):
            texts = [str(x) for x in r["choices"]["text"]]
            src_labels = [str(x) for x in r["choices"]["label"]]
            gi = normalize_answer_key(r["answerKey"], src_labels, len(texts))
            raw.append({
                "id": str(r.get("id", i)),
                "question": str(r["question"]),
                "choices": texts,
                "gold_index": gi,
            })

    elif name == "openbookqa":
        ds = load_dataset("allenai/openbookqa", "main", split="validation")
        raw = []
        for i, r in enumerate(ds):
            texts = [str(x) for x in r["choices"]["text"]]
            src_labels = [str(x) for x in r["choices"]["label"]]
            gi = normalize_answer_key(r["answerKey"], src_labels, len(texts))
            raw.append({
                "id": str(r.get("id", i)),
                "question": str(r["question_stem"]),
                "choices": texts,
                "gold_index": gi,
            })

    elif name == "sciq":
        ds = load_dataset("allenai/sciq", split="validation")
        raw = []
        for i, r in enumerate(ds):
            correct = str(r["correct_answer"])
            options = [
                correct,
                str(r["distractor1"]),
                str(r["distractor2"]),
                str(r["distractor3"]),
            ]
            # Shuffle so the correct answer is not always A.
            local = random.Random(seed * 1000003 + i)
            perm = list(range(4))
            local.shuffle(perm)
            choices = [options[j] for j in perm]
            gi = perm.index(0)
            raw.append({
                "id": f"sciq-{i}",
                "question": str(r["question"]),
                "choices": choices,
                "gold_index": gi,
            })

    elif name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        raw = []
        for i, r in enumerate(ds):
            choices = [str(x) for x in r["endings"]]
            gi = int(r["label"])
            raw.append({
                "id": str(r.get("ind", i)),
                "question": str(r["ctx"]),
                "choices": choices,
                "gold_index": gi,
            })
    else:
        raise ValueError(name)

    # Keep only genuine 4-choice examples to make the settings comparable.
    raw = [r for r in raw if len(r["choices"]) == 4 and 0 <= r["gold_index"] < 4]
    rng.shuffle(raw)
    if max_samples is not None:
        raw = raw[:max_samples]

    for r in raw:
        r["labels"] = list("ABCD")
        r["gold"] = r["labels"][r["gold_index"]]
    return raw


def format_prompt(dataset: str, row: dict) -> str:
    opts = "\n".join(f"{lab}: {txt}" for lab, txt in zip(row["labels"], row["choices"]))
    if dataset == "hellaswag":
        intro = (
            "Choose the single most plausible continuation of the context. "
            "Consider commonsense and temporal coherence."
        )
        qlabel = "Context"
    else:
        intro = (
            "Choose the single best answer to the multiple-choice question. "
            "Consider all four options carefully."
        )
        qlabel = "Question"
    return (
        f"{intro}\n"
        "Respond with exactly one letter: A, B, C, or D. Do not explain.\n\n"
        f"{qlabel}: {row['question']}\n\n"
        f"{opts}\n\n"
        "Answer:"
    )


def parse_choice(text: str, labels: list[str]) -> str | None:
    s = str(text or "").strip()
    allowed = "".join(re.escape(x) for x in labels)
    patterns = [
        rf"(?i)^\s*[\(\[]?([{allowed}])[\)\]\.\:\-\s]?",
        rf"(?i)\banswer\s*(?:is|:)?\s*[\(\[]?([{allowed}])[\)\]]?\b",
        rf"(?i)\b(?:option|choice)\s*[\(\[]?([{allowed}])[\)\]]?\b",
        rf"(?i)\b([{allowed}])\b",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            c = m.group(1).upper()
            if c in labels:
                return c
    return None


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    counts = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        counts[i, : int(remainder[i])] += 1
    return counts


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    z = logits.float()
    logp = F.log_softmax(z, dim=-1)
    p = logp.exp()
    h = -(p * logp).sum(dim=-1)
    return (h / math.log(z.size(-1))).clamp(0.0, 1.0)


def safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def safe_std(x):
    return float(np.std(x)) if len(x) else 0.0


def safe_max(x):
    return float(np.max(x)) if len(x) else 0.0


def safe_q(x, q):
    return float(np.quantile(x, q)) if len(x) else 0.0


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=float)
    order = np.argsort(values)
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    return rank / float(n - 1)


@torch.inference_mode()
def generate_with_rcc_trace(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    mask_id: int,
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    if prompt_ids.size(0) != 1:
        raise ValueError("batch size must be 1")
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by gen_length // block_length")

    device = model.device
    prompt_len = prompt_ids.size(1)
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids
    full_attn = torch.cat([
        attention_mask,
        torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device),
    ], dim=-1)

    steps_per_block = steps // num_blocks
    histories: List[List[float]] = [[] for _ in range(gen_length)]

    jumps_all, jumps_late = [], []
    volatilities, overshoots = [], []
    sel_pct_all, sel_pct_late = [], []
    sel_gap_all, sel_gap_late = [], []
    sel_overlap_all, sel_overlap_late = [], []
    sel_regret_all, sel_regret_late = [], []

    global_step = 0
    for block_idx in range(num_blocks):
        bs = prompt_len + block_idx * block_length
        be = prompt_len + (block_idx + 1) * block_length
        rel0 = block_idx * block_length
        block_mask = x[:, bs:be].eq(mask_id)
        n_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

        for local_step in range(steps_per_block):
            unresolved = torch.nonzero(x[0, bs:be].eq(mask_id), as_tuple=False).flatten()
            logits_full = model(x, attention_mask=full_attn).logits
            logits = logits_full[:, bs:be, :]
            noisy = add_gumbel_noise(logits, temperature)
            pred = noisy.argmax(dim=-1)[0]
            probs = F.softmax(logits.float(), dim=-1)
            conf = probs.gather(-1, pred.view(1, -1, 1)).squeeze(-1)[0]
            ent = normalized_entropy(logits)[0]

            for j in unresolved.tolist():
                histories[rel0 + j].append(float(ent[j].item()))

            k = int(n_transfer[0, local_step].item())
            if k <= 0 or len(unresolved) == 0:
                global_step += 1
                continue
            k = min(k, len(unresolved))
            chosen = unresolved[torch.topk(conf[unresolved], k=k).indices]

            u = unresolved
            u_ent = ent[u].float().cpu().numpy()
            pct = percentile_ranks(u_ent)
            index_lookup = {int(pos): idx for idx, pos in enumerate(u.tolist())}
            chosen_idx = [index_lookup[int(pos)] for pos in chosen.tolist()]
            chosen_ent = u_ent[chosen_idx]

            sp = float(np.mean(pct[chosen_idx]))
            remain = np.ones(len(u_ent), dtype=bool)
            remain[chosen_idx] = False
            gap = float(np.mean(chosen_ent) - np.mean(u_ent[remain])) if remain.any() else 0.0
            greedy_idx = np.argsort(u_ent)[:k]
            overlap = len(set(chosen_idx).intersection(set(greedy_idx))) / float(k)
            regret = (float(np.mean(chosen_ent) - np.mean(u_ent[greedy_idx])) / (float(np.std(u_ent)) + 1e-8))

            sel_pct_all.append(sp)
            sel_gap_all.append(gap)
            sel_overlap_all.append(float(overlap))
            sel_regret_all.append(regret)

            late = global_step >= int(math.floor(2 * steps / 3))
            if late:
                sel_pct_late.append(sp)
                sel_gap_late.append(gap)
                sel_overlap_late.append(float(overlap))
                sel_regret_late.append(regret)

            for j in chosen.tolist():
                hist = histories[rel0 + int(j)]
                if len(hist) >= 2:
                    jump = float(hist[-2] - hist[-1])
                    diffs = np.diff(np.asarray(hist, dtype=float))
                    volatility = float(np.std(diffs)) if len(diffs) else 0.0
                    overshoot = float(np.max(hist) - hist[-1])
                else:
                    jump = volatility = overshoot = 0.0
                jumps_all.append(jump)
                volatilities.append(volatility)
                overshoots.append(overshoot)
                if late:
                    jumps_late.append(jump)

            x[0, bs + chosen] = pred[chosen]
            global_step += 1

    text = tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)
    feats = {
        "late_jump_mean": safe_mean(jumps_late),
        "late_jump_q90": safe_q(jumps_late, 0.90),
        "all_jump_mean": safe_mean(jumps_all),
        "all_jump_std": safe_std(jumps_all),
        "all_jump_max": safe_max(jumps_all),
        "all_volatility_max": safe_max(volatilities),
        "all_volatility_std": safe_std(volatilities),
        "all_peak_overshoot_max": safe_max(overshoots),
        "sel_percentile_mean": safe_mean(sel_pct_all),
        "sel_percentile_late": safe_mean(sel_pct_late),
        "sel_gap_mean": safe_mean(sel_gap_all),
        "sel_gap_late": safe_mean(sel_gap_late),
        "sel_overlap_mean": safe_mean(sel_overlap_all),
        "sel_overlap_late": safe_mean(sel_overlap_late),
        "sel_regret_norm_mean": safe_mean(sel_regret_all),
        "sel_regret_norm_late": safe_mean(sel_regret_late),
    }
    vec = np.asarray([feats[n] for n in FEATURE_NAMES], dtype=np.float32)
    return text, vec, feats


def load_model(model_dir: str, device: str):
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype).to(dev).eval()
    return model, tokenizer


def encode_prompt(tokenizer, text: str, device: torch.device):
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tokenizer(chat, add_special_tokens=False, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def selective_curve(y_correct: np.ndarray, risk: np.ndarray) -> pd.DataFrame:
    base_acc = float(y_correct.mean())
    base_err = 1.0 - base_acc
    order = np.argsort(-risk)
    rows = []
    for rate in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        nrej = int(round(rate * len(y_correct)))
        keep = np.ones(len(y_correct), dtype=bool)
        if nrej:
            keep[order[:nrej]] = False
        acc = float(y_correct[keep].mean()) if keep.any() else np.nan
        err = 1.0 - acc if np.isfinite(acc) else np.nan
        rej_acc = float(y_correct[~keep].mean()) if (~keep).any() else np.nan
        rel = ((base_err - err) / base_err) if base_err > 0 and np.isfinite(err) else np.nan
        rows.append({
            "abstain_rate": rate,
            "coverage": float(keep.mean()),
            "answered_n": int(keep.sum()),
            "abstained_n": int((~keep).sum()),
            "answered_accuracy": acc,
            "answered_error_rate": err,
            "relative_error_reduction": rel,
            "abstained_subset_accuracy": rej_acc,
        })
    return pd.DataFrame(rows)


def evaluate_dataset(name: str, rows: list[dict], model, tokenizer, args, root: Path) -> dict:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "generations.jsonl"
    existing = {str(r["id"]): r for r in load_jsonl(cache_path)}

    start_time = time.perf_counter()
    for row in tqdm(rows, desc=f"{name}: RCC trajectories", unit="sample"):
        rid = str(row["id"])
        if rid in existing:
            continue
        prompt = format_prompt(name, row)
        ids, am = encode_prompt(tokenizer, prompt, model.device)
        text, vec, featdict = generate_with_rcc_trace(
            model,
            tokenizer,
            ids,
            am,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            mask_id=int(getattr(model.config, "mask_token_id", args.mask_id)),
        )
        pred = parse_choice(text, row["labels"])
        correct = int(pred == row["gold"])
        rec = {
            "id": rid,
            "question": row["question"],
            "choices": row["choices"],
            "gold": row["gold"],
            "prediction": pred,
            "correct": correct,
            "error": 1 - correct,
            "output_text": text,
            **featdict,
        }
        dump_jsonl_row(cache_path, rec)
        existing[rid] = rec

    elapsed = time.perf_counter() - start_time
    records = [existing[str(r["id"])] for r in rows if str(r["id"]) in existing]
    if len(records) != len(rows):
        raise RuntimeError(f"{name}: only {len(records)}/{len(rows)} completed records")

    df = pd.DataFrame(records)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_error = df["error"].to_numpy(dtype=int)
    y_correct = df["correct"].to_numpy(dtype=int)

    if len(np.unique(y_error)) < 2:
        raise RuntimeError(f"{name}: all examples have the same correctness label")

    ids = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        ids,
        train_size=args.train_fraction,
        random_state=args.seed,
        shuffle=True,
        stratify=y_error,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=args.seed),
    )
    fit_t0 = time.perf_counter()
    clf.fit(X[train_idx], y_error[train_idx])
    fit_seconds = time.perf_counter() - fit_t0

    train_risk = clf.predict_proba(X[train_idx])[:, 1]
    test_risk = clf.predict_proba(X[test_idx])[:, 1]
    test_error = y_error[test_idx]
    test_correct = y_correct[test_idx]

    auroc = float(roc_auc_score(test_error, test_risk))
    auprc = float(average_precision_score(test_error, test_risk))
    curve = selective_curve(test_correct, test_risk)
    curve.to_csv(out / "heldout_selective_curve.csv", index=False)

    q25 = curve[np.isclose(curve["abstain_rate"], 0.25)].iloc[0]
    base_all = float(y_correct.mean())
    base_test = float(test_correct.mean())

    # Fixed train-calibrated threshold (secondary robustness statistic).
    train_thr = float(np.quantile(train_risk, 0.75))
    keep_fixed = test_risk < train_thr
    fixed_cov = float(keep_fixed.mean())
    fixed_acc = float(test_correct[keep_fixed].mean()) if keep_fixed.any() else float("nan")

    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[test_idx, "split"] = "test"
    df["rcc_risk"] = np.nan
    df.loc[train_idx, "rcc_risk"] = train_risk
    df.loc[test_idx, "rcc_risk"] = test_risk
    df.to_csv(out / "results_with_risk.csv", index=False)

    summary = {
        "dataset": name,
        "n": int(len(df)),
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "baseline_accuracy_all": base_all,
        "baseline_accuracy_test": base_test,
        "rcc_test_auroc": auroc,
        "rcc_test_auprc": auprc,
        "top25_abstention": {
            "coverage": float(q25["coverage"]),
            "answered_accuracy": float(q25["answered_accuracy"]),
            "relative_error_reduction": float(q25["relative_error_reduction"]),
            "abstained_subset_accuracy": float(q25["abstained_subset_accuracy"]),
        },
        "train_threshold_applied_to_test": {
            "risk_threshold": train_thr,
            "coverage": fixed_cov,
            "answered_accuracy": fixed_acc,
        },
        "generation_seconds_this_run": elapsed,
        "detector_fit_seconds": fit_seconds,
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(curve["coverage"] * 100, curve["answered_accuracy"] * 100, marker="o")
    ax.axhline(base_test * 100, linestyle="--", linewidth=1, label=f"Baseline {100*base_test:.1f}%")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy on answered samples (%)")
    ax.set_title(f"{name}: RCC selective answering")
    ax.grid(True, alpha=0.25)
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "selective_accuracy_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=DATASET_CHOICES)
    ap.add_argument("--model_dir", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--output_dir", default="outputs/rcc_mcq_sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--gen_length", type=int, default=64)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--train_fraction", type=float, default=0.50)
    ap.add_argument("--mask_id", type=int, default=MASK_ID_DEFAULT)
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("MULTI-DATASET RCC SELECTIVE MITIGATION SWEEP")
    print("=" * 112)
    print(f"datasets       : {', '.join(args.datasets)}")
    print(f"samples/dataset: {args.max_samples}")
    print(f"generation     : steps={args.steps}, gen={args.gen_length}, block={args.block_length}")
    print("evaluation     : exact MCQ ground truth; no LLM judge")
    print("resume         : enabled (existing generations.jsonl are reused)")

    model, tokenizer = load_model(args.model_dir, args.device)

    summaries = []
    for name in args.datasets:
        rows = load_mcq_rows(name, args.max_samples, args.seed)
        print("\n" + "-" * 112)
        print(f"{name}: loaded {len(rows)} four-choice examples")
        try:
            summary = evaluate_dataset(name, rows, model, tokenizer, args, root)
            summaries.append(summary)
            t = summary["top25_abstention"]
            print(
                f"{name}: base(test)={100*summary['baseline_accuracy_test']:.2f}%  "
                f"AUROC={100*summary['rcc_test_auroc']:.2f}  "
                f"@75%cov={100*t['answered_accuracy']:.2f}%  "
                f"rel.err.red={100*t['relative_error_reduction']:+.2f}%"
            )
        except Exception as exc:
            print(f"{name}: FAILED: {exc}")
            summaries.append({"dataset": name, "error": str(exc)})

    good = [s for s in summaries if "error" not in s]
    flat = []
    for s in good:
        t = s["top25_abstention"]
        flat.append({
            "dataset": s["dataset"],
            "n": s["n"],
            "baseline_accuracy_test": s["baseline_accuracy_test"],
            "rcc_test_auroc": s["rcc_test_auroc"],
            "rcc_test_auprc": s["rcc_test_auprc"],
            "coverage": t["coverage"],
            "selective_accuracy": t["answered_accuracy"],
            "relative_error_reduction": t["relative_error_reduction"],
            "abstained_subset_accuracy": t["abstained_subset_accuracy"],
            "fixed_threshold_coverage": s["train_threshold_applied_to_test"]["coverage"],
            "fixed_threshold_accuracy": s["train_threshold_applied_to_test"]["answered_accuracy"],
        })

    rank = pd.DataFrame(flat)
    if len(rank):
        rank = rank.sort_values(["relative_error_reduction", "rcc_test_auroc"], ascending=False)
        rank.to_csv(root / "pilot_summary.csv", index=False)
        positive = rank[rank["relative_error_reduction"] > 0]
        recommended = positive.head(2) if len(positive) else rank.head(2)
        recommended.to_csv(root / "recommended_two.csv", index=False)

        print("\n" + "=" * 112)
        print("PILOT RANKING (held-out test; top-25% risk abstention)")
        print("=" * 112)
        for _, r in rank.iterrows():
            print(
                f"{r['dataset']:12s} base={100*r['baseline_accuracy_test']:6.2f}%  "
                f"RCC-AUROC={100*r['rcc_test_auroc']:6.2f}  "
                f"sel@75={100*r['selective_accuracy']:6.2f}%  "
                f"rel.err.red={100*r['relative_error_reduction']:+7.2f}%"
            )
        print("\nRecommended two for main-text emphasis (pilot selection):")
        for _, r in recommended.iterrows():
            print(f"  - {r['dataset']}  (relative error reduction {100*r['relative_error_reduction']:+.2f}%)")
        print("\nKeep pilot_summary.csv in the experiment record; do not silently discard failed datasets.")

    (root / "all_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSaved: {root / 'pilot_summary.csv'}")
    print(f"Saved: {root / 'recommended_two.csv'}")
    print(f"Saved: {root / 'all_summaries.json'}")


if __name__ == "__main__":
    main()
