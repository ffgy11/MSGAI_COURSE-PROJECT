from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rcc_dllm.generation.llada import (
    entropy_from_logits,
    masked_mean,
    masked_sum,
    remove_special_token_entropy,
)


_ORIGINAL_MULTINOMIAL = torch.multinomial


def _sanitize_probabilities(probs: torch.Tensor) -> torch.Tensor:
    probs = torch.nan_to_num(probs.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if probs.ndim == 1:
        total = probs.sum()
        if total <= 0:
            return torch.full_like(probs, 1.0 / probs.numel())
        return probs / total
    total = probs.sum(dim=-1, keepdim=True)
    zero = total <= 0
    if zero.any():
        probs = probs.clone()
        probs[zero.squeeze(-1)] = 1.0
        total = probs.sum(dim=-1, keepdim=True)
    return probs / total


def _safe_multinomial(probs, num_samples, replacement=False, *, generator=None, out=None):
    return _ORIGINAL_MULTINOMIAL(
        _sanitize_probabilities(probs),
        num_samples,
        replacement,
        generator=generator,
        out=out,
    )


def ensure_safe_multinomial() -> None:
    if not getattr(torch.multinomial, "_tre_dream_safe", False):
        _safe_multinomial._tre_dream_safe = True  # type: ignore[attr-defined]
        torch.multinomial = _safe_multinomial  # type: ignore[assignment]


def prepare_dream_attention(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_length: int,
) -> tuple[torch.Tensor | str, torch.Tensor | None]:
    if attention_mask is not None and torch.any(attention_mask == 0):
        attention_mask = F.pad(
            attention_mask.to(input_ids.device),
            (0, max_length - input_ids.shape[1]),
            value=1.0,
        )
        token_index = attention_mask.long().cumsum(-1) - 1
        token_index.masked_fill_(attention_mask == 0, 1)
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        return attention_mask, token_index
    return "full", None


@torch.no_grad()
def generate_dream_with_entropy(
    model,
    prompt: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    steps: int = 128,
    gen_length: int = 128,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    alg: str = "origin",
    alg_temp: float | None = 0.0,
    eps: float = 1e-3,
    mask_id: int | None = None,
    ban_eos_first_tokens: int = 0,
    special_token_ids: list[int] | None = None,
) -> tuple[torch.Tensor, list[np.ndarray], dict, dict]:
    """Generate with Dream and record entropy/reveal trajectories.

    The output format matches the LLaDA generator: an entropy matrix per denoising
    step and per-step newly revealed absolute positions. These saved traces can be
    scored by the same TRE implementation.
    """
    ensure_safe_multinomial()

    device = prompt.device
    batch_size = int(prompt.shape[0])
    prompt_len = int(prompt.shape[1])
    total_length = prompt_len + int(gen_length)
    mask_id = int(mask_id if mask_id is not None else model.config.mask_token_id)
    eos_id = getattr(model.config, "eos_token_id", None)
    special_ids = {int(x) for x in (special_token_ids or [])}

    entropy_steps: list[np.ndarray] = []
    remaining_entropy_before_mean = np.full((steps, batch_size), np.nan, dtype=np.float32)
    remaining_entropy_before_sum = np.full((steps, batch_size), np.nan, dtype=np.float32)
    revealed_counts = np.zeros((steps, batch_size), dtype=np.int32)
    revealed_positions: list[list[np.ndarray]] = []
    revealed_token_ids: list[list[np.ndarray]] = []
    revealed_token_entropies: list[list[np.ndarray]] = []
    revealed_token_entropy_mean = np.full((steps, batch_size), np.nan, dtype=np.float32)

    prepare_dream_attention(prompt, attention_mask, total_length)
    previous_tokens = F.pad(prompt, (0, total_length - prompt_len), value=mask_id)
    current_entropy: torch.Tensor | None = None

    progress = tqdm(total=steps, desc="dream_steps", leave=False)
    last_step = -1

    def logits_hook(step: int, x: torch.Tensor, logits: torch.Tensor):
        nonlocal current_entropy
        if logits is None:
            return logits

        aligned_logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        entropy = entropy_from_logits(aligned_logits)
        current_entropy = entropy
        mask_before = previous_tokens == mask_id
        entropy_steps.append(entropy.detach().cpu().numpy().astype(np.float32))
        remaining_entropy_before_mean[step] = masked_mean(entropy, mask_before).detach().cpu().numpy().astype(np.float32)
        remaining_entropy_before_sum[step] = masked_sum(entropy, mask_before).detach().cpu().numpy().astype(np.float32)

        if ban_eos_first_tokens > 0 and eos_id is not None:
            lo = prompt_len
            hi = min(prompt_len + int(ban_eos_first_tokens), logits.shape[1])
            logits[:, lo:hi, int(eos_id)] = torch.finfo(logits.dtype).min
        return logits

    def tokens_hook(step: int | None, x: torch.Tensor, logits: torch.Tensor | None):
        nonlocal previous_tokens, last_step
        if step is None:
            return x

        if step > last_step:
            progress.update(step - last_step)
            last_step = step

        newly_revealed = (previous_tokens == mask_id) & (x != mask_id)
        step_positions: list[np.ndarray] = []
        step_token_ids: list[np.ndarray] = []
        step_entropies: list[np.ndarray] = []

        for batch_index in range(batch_size):
            pos = torch.nonzero(newly_revealed[batch_index], as_tuple=False).squeeze(-1)
            ids = x[batch_index, pos] if pos.numel() else torch.empty((0,), device=device, dtype=torch.long)
            if pos.numel() and current_entropy is not None:
                token_entropy = remove_special_token_entropy(ids, current_entropy[batch_index, pos], special_ids)
                token_entropy_np = token_entropy.detach().cpu().numpy().astype(np.float32)
                revealed_token_entropy_mean[step, batch_index] = float(token_entropy_np.mean()) if token_entropy_np.size else np.nan
            else:
                token_entropy_np = np.empty((0,), dtype=np.float32)

            revealed_counts[step, batch_index] = int(pos.numel())
            step_positions.append(pos.detach().cpu().numpy().astype(np.int32))
            step_token_ids.append(ids.detach().cpu().numpy().astype(np.int32))
            step_entropies.append(token_entropy_np)

        revealed_positions.append(step_positions)
        revealed_token_ids.append(step_token_ids)
        revealed_token_entropies.append(step_entropies)
        previous_tokens = x.clone()
        return x

    try:
        output = model.diffusion_generate(
            prompt,
            attention_mask=attention_mask,
            max_new_tokens=int(gen_length),
            output_history=True,
            return_dict_in_generate=True,
            steps=int(steps),
            temperature=float(temperature),
            top_p=top_p,
            top_k=top_k,
            alg=str(alg),
            alg_temp=float(alg_temp) if alg_temp is not None else None,
            eps=float(eps),
            generation_tokens_hook_func=tokens_hook,
            generation_logits_hook_func=logits_hook,
        )
    finally:
        progress.close()

    np.nan_to_num(remaining_entropy_before_mean, nan=0.0, copy=False)
    np.nan_to_num(remaining_entropy_before_sum, nan=0.0, copy=False)

    meta = {
        "total_steps": int(steps),
        "prompt_len": int(prompt_len),
        "gen_length": int(gen_length),
        "seq_len": int(total_length),
        "model_family": "dream",
        "dream_alg": str(alg),
        "dream_eps": float(eps),
        "dream_temperature": float(temperature),
        "dream_top_p": None if top_p is None else float(top_p),
        "dream_top_k": None if top_k is None else int(top_k),
        "dream_alg_temp": None if alg_temp is None else float(alg_temp),
    }
    step_trace = {
        "remaining_entropy_before_mean": remaining_entropy_before_mean,
        "remaining_entropy_before_sum": remaining_entropy_before_sum,
        "revealed_positions": revealed_positions,
        "revealed_token_ids": revealed_token_ids,
        "revealed_token_entropies": revealed_token_entropies,
        "revealed_token_entropy_mean": revealed_token_entropy_mean,
        "revealed_counts": revealed_counts,
    }
    return output.sequences, entropy_steps, meta, step_trace
