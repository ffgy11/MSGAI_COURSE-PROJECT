from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Trace:
    entropy: np.ndarray
    prompt_len: int
    gen_length: int
    revealed_counts: np.ndarray
    revealed_positions: np.ndarray


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_label(row: dict[str, Any]) -> int | None:
    value = row.get("hallucination")
    if isinstance(value, dict):
        label = value.get("label")
    elif isinstance(value, str):
        label = value
    else:
        label = row.get("label")

    if not isinstance(label, str):
        return None
    label = label.strip().lower().replace("-", "_")
    if label == "hallucination":
        return 1
    if label == "non_hallucination":
        return 0
    return None


def _read_scalar(npz: Any, names: list[str], default: int = 0) -> int:
    for name in names:
        if name in npz:
            arr = np.asarray(npz[name]).reshape(-1)
            if arr.size:
                return int(arr[0])
    return int(default)


def _resolve_entropy_file(row: dict[str, Any], manifest_path: Path) -> Path | None:
    raw = row.get("entropy_file") or row.get("npz_path") or row.get("entropy_path")
    if not raw:
        return None

    path = Path(str(raw))
    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        base = manifest_path.parent
        candidates = [
            base / path,
            base / "entropy" / path.name,
            base / path.name,
            path,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_trace(path: str | Path) -> Trace:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        entropy = np.asarray(z["entropy"], dtype=np.float64)
        if entropy.ndim != 2:
            raise ValueError(f"entropy must be 2D, got shape {entropy.shape}")

        prompt_len = _read_scalar(z, ["prompt_len", "prompt_length"], default=0)
        gen_length = _read_scalar(z, ["gen_length", "generation_length"], default=entropy.shape[1] - prompt_len)
        total_steps = _read_scalar(z, ["total_steps"], default=entropy.shape[0])

        if entropy.shape[0] != total_steps and entropy.shape[1] == total_steps:
            entropy = entropy.T

        revealed_counts = np.asarray(z["revealed_counts"], dtype=np.int64).reshape(-1)
        revealed_positions = np.asarray(z["revealed_positions"], dtype=np.int64)

    return Trace(
        entropy=entropy,
        prompt_len=int(prompt_len),
        gen_length=int(gen_length),
        revealed_counts=revealed_counts,
        revealed_positions=revealed_positions,
    )


def _generation_entropy(trace: Trace) -> np.ndarray:
    start = trace.prompt_len
    end = start + trace.gen_length
    if trace.entropy.shape[1] >= end:
        return trace.entropy[:, start:end]
    return trace.entropy[:, : trace.gen_length]


def _infer_position_mode(trace: Trace, gen_length: int) -> str:
    counts = np.asarray(trace.revealed_counts, dtype=np.int64).reshape(-1)
    positions = []
    for step, count in enumerate(counts[: trace.revealed_positions.shape[0]]):
        count = int(count)
        if count > 0:
            positions.append(np.asarray(trace.revealed_positions[step, :count], dtype=np.int64))
    if not positions:
        return "relative"
    all_pos = np.concatenate(positions)
    if all_pos.size == 0:
        return "relative"
    if np.any(all_pos >= gen_length) or np.nanmedian(all_pos) >= trace.prompt_len:
        return "absolute"
    return "relative"


def _relative_new_positions(trace: Trace, step: int, gen_length: int, position_mode: str) -> np.ndarray:
    count = int(trace.revealed_counts[step]) if step < len(trace.revealed_counts) else 0
    if count <= 0:
        return np.empty((0,), dtype=np.int64)

    pos = np.asarray(trace.revealed_positions[step, :count], dtype=np.int64).reshape(-1)
    pos = pos[pos >= 0]
    if pos.size == 0:
        return pos

    if position_mode == "absolute":
        pos = pos - int(trace.prompt_len)

    return pos[(pos >= 0) & (pos < gen_length)]


def compute_tre_from_trace(trace: Trace, empty_value: float = 0.0) -> float:
    entropy = _generation_entropy(trace)
    total_steps = min(entropy.shape[0], len(trace.revealed_counts), trace.revealed_positions.shape[0])
    if total_steps <= 0:
        return float("nan")

    gen_length = entropy.shape[1]
    revealed = np.zeros(gen_length, dtype=bool)
    step_mass = np.zeros(total_steps, dtype=np.float64)
    position_mode = _infer_position_mode(trace, gen_length)

    for step in range(total_steps):
        pos = _relative_new_positions(trace, step, gen_length, position_mode)
        if pos.size == 0:
            step_mass[step] = empty_value
            continue

        pos = np.asarray(sorted(set(int(x) for x in pos.tolist() if not revealed[int(x)])), dtype=np.int64)
        if pos.size == 0:
            step_mass[step] = empty_value
            continue

        values = entropy[step, pos]
        values = values[np.isfinite(values)]
        step_mass[step] = float(np.sum(values)) if values.size else float(empty_value)
        revealed[pos] = True

    weights = np.arange(1, total_steps + 1, dtype=np.float64) / float(total_steps)
    return float(np.sum(weights * step_mass))


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average_ties(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))



def metric_bundle(labels: list[int], scores: list[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0:
        return {}
    return {
        "n": int(len(y)),
        "n_hallucination": int((y == 1).sum()),
        "n_non_hallucination": int((y == 0).sum()),
        "auroc": auroc(y, s),
    }
def score_manifest(manifest_path: str | Path, empty_value: float = 0.0) -> tuple[list[dict[str, Any]], dict[str, float]]:
    manifest_path = Path(manifest_path)
    rows = load_jsonl(manifest_path)
    score_rows: list[dict[str, Any]] = []
    labels: list[int] = []
    scores: list[float] = []

    for index, row in enumerate(rows):
        entropy_path = _resolve_entropy_file(row, manifest_path)
        score = float("nan")
        if entropy_path is not None:
            score = compute_tre_from_trace(load_trace(entropy_path), empty_value=empty_value)

        label = parse_label(row)
        out = {
            "index": row.get("index", index),
            "question_id": row.get("question_id", row.get("qid", "")),
            "tre_score": score,
            "label": "" if label is None else int(label),
            "entropy_file": "" if entropy_path is None else str(entropy_path),
        }
        score_rows.append(out)

        if label is not None and np.isfinite(score):
            labels.append(label)
            scores.append(score)

    summary = metric_bundle(labels, scores) if labels else {}
    return score_rows, summary
