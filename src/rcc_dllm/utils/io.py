from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def safe_name(value: str, max_len: int = 160) -> str:
    text = str(value)
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
    return out[:max_len] or "sample"


def save_entropy_trace(
    path: str | Path,
    *,
    entropy: np.ndarray,
    meta: dict[str, Any],
    step_trace: dict[str, Any],
    batch_index: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    counts = step_trace["revealed_counts"][:, batch_index].astype(np.int32)
    width = max(int(counts.max()) if counts.size else 0, 1)
    positions = np.full((counts.shape[0], width), -1, dtype=np.int32)
    token_ids = np.full((counts.shape[0], width), -1, dtype=np.int32)
    token_entropies = np.full((counts.shape[0], width), np.nan, dtype=np.float32)

    for step in range(counts.shape[0]):
        n = int(counts[step])
        if n <= 0:
            continue
        positions[step, :n] = step_trace["revealed_positions"][step][batch_index][:n]
        token_ids[step, :n] = step_trace["revealed_token_ids"][step][batch_index][:n]
        token_entropies[step, :n] = step_trace["revealed_token_entropies"][step][batch_index][:n]

    np.savez_compressed(
        path,
        entropy=entropy.astype(np.float32),
        prompt_len=np.array([int(meta["prompt_len"])], dtype=np.int32),
        gen_length=np.array([int(meta["gen_length"])], dtype=np.int32),
        total_steps=np.array([int(meta["total_steps"])], dtype=np.int32),
        revealed_counts=counts,
        revealed_positions=positions,
        revealed_token_ids=token_ids,
        revealed_token_entropies=token_entropies,
        revealed_token_entropy_mean=step_trace["revealed_token_entropy_mean"][:, batch_index].astype(np.float32),
        remaining_entropy_before_mean=step_trace["remaining_entropy_before_mean"][:, batch_index].astype(np.float32),
        remaining_entropy_before_sum=step_trace["remaining_entropy_before_sum"][:, batch_index].astype(np.float32),
    )


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
