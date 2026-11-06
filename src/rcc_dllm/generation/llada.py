from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    logits = logits.float()
    noise = torch.rand_like(logits)
    return logits / ((-torch.log(noise)) ** temperature)


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        step_ids = torch.arange(steps, device=mask_index.device)
        out[step_ids.unsqueeze(0) < remainder] += 1
    return out.long()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum(dim=1)
    out = (values * mask_f).sum(dim=1) / torch.clamp(denom, min=1.0)
    return torch.where(denom > 0, out, torch.full_like(out, float("nan")))


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.float()).sum(dim=1)


def remove_special_token_entropy(
    token_ids: torch.Tensor,
    token_entropies: torch.Tensor,
    special_token_ids: set[int] | None,
) -> torch.Tensor:
    if not special_token_ids or token_entropies.numel() == 0:
        return token_entropies
    special_ids = torch.tensor(sorted(special_token_ids), device=token_ids.device, dtype=token_ids.dtype)
    special_mask = (token_ids.unsqueeze(-1) == special_ids).any(dim=-1)
    return torch.where(special_mask, torch.zeros_like(token_entropies), token_entropies)


@torch.no_grad()
def generate_with_entropy(
    model,
    prompt: torch.Tensor,
    *,
    steps: int = 128,
    gen_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    mask_id: int = 126336,
    special_token_ids: list[int] | None = None,
) -> tuple[torch.Tensor, list[np.ndarray], dict, dict]:
    device = prompt.device
    batch_size = int(prompt.shape[0])
    sequence_length = int(prompt.shape[1] + gen_length)
    special_ids = {int(x) for x in (special_token_ids or [])}

    entropy_steps: list[np.ndarray] = []
    revealed_counts = np.zeros((steps, batch_size), dtype=np.int32)
    revealed_token_entropy_mean = np.full((steps, batch_size), np.nan, dtype=np.float32)
    remaining_entropy_before_mean = np.full((steps, batch_size), np.nan, dtype=np.float32)
    remaining_entropy_before_sum = np.full((steps, batch_size), np.nan, dtype=np.float32)
    revealed_positions: list[list[np.ndarray]] = []
    revealed_token_ids: list[list[np.ndarray]] = []
    revealed_token_entropies: list[list[np.ndarray]] = []

    x = torch.full((batch_size, sequence_length), mask_id, dtype=torch.long, device=device)
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id
    num_transfer_tokens = get_num_transfer_tokens(x == mask_id, steps)

    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        for step in tqdm(range(steps), desc="denoising", leave=False):
            mask_index = x == mask_id

            if cfg_scale > 0.0:
                uncond = x.clone()
                uncond[prompt_index] = mask_id
                logits, uncond_logits = torch.chunk(model(torch.cat([x, uncond], dim=0)).logits, 2, dim=0)
                logits = uncond_logits + (cfg_scale + 1.0) * (logits - uncond_logits)
            else:
                logits = model(x).logits

            entropy = entropy_from_logits(logits)
            entropy_steps.append(entropy.cpu().numpy().astype(np.float32))
            remaining_entropy_before_mean[step] = masked_mean(entropy, mask_index).cpu().numpy().astype(np.float32)
            remaining_entropy_before_sum[step] = masked_sum(entropy, mask_index).cpu().numpy().astype(np.float32)

            noisy_logits = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(noisy_logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            confidence = torch.where(mask_index, confidence, torch.tensor(-np.inf, device=device))
            x0 = torch.where(mask_index, x0, x)

            step_positions: list[np.ndarray] = []
            step_ids: list[np.ndarray] = []
            step_entropies: list[np.ndarray] = []

            for b in range(batch_size):
                n_transfer = int(num_transfer_tokens[b, step].item())
                if n_transfer > 0:
                    _, selected = torch.topk(confidence[b], k=n_transfer)
                    selected_ids = x0[b, selected]
                    selected_entropy = remove_special_token_entropy(
                        selected_ids,
                        entropy[b, selected],
                        special_ids,
                    )
                    x[b, selected] = selected_ids

                    pos_np = selected.detach().cpu().numpy().astype(np.int32)
                    ids_np = selected_ids.detach().cpu().numpy().astype(np.int32)
                    ent_np = selected_entropy.detach().cpu().numpy().astype(np.float32)
                    revealed_counts[step, b] = n_transfer
                    revealed_token_entropy_mean[step, b] = float(np.mean(ent_np))
                else:
                    pos_np = np.empty((0,), dtype=np.int32)
                    ids_np = np.empty((0,), dtype=np.int32)
                    ent_np = np.empty((0,), dtype=np.float32)

                step_positions.append(pos_np)
                step_ids.append(ids_np)
                step_entropies.append(ent_np)

            revealed_positions.append(step_positions)
            revealed_token_ids.append(step_ids)
            revealed_token_entropies.append(step_entropies)

    meta = {
        "total_steps": int(steps),
        "prompt_len": int(prompt.shape[1]),
        "gen_length": int(gen_length),
        "seq_len": int(sequence_length),
        "mask_id": int(mask_id),
    }
    trace = {
        "remaining_entropy_before_mean": remaining_entropy_before_mean,
        "remaining_entropy_before_sum": remaining_entropy_before_sum,
        "revealed_positions": revealed_positions,
        "revealed_token_ids": revealed_token_ids,
        "revealed_token_entropies": revealed_token_entropies,
        "revealed_token_entropy_mean": revealed_token_entropy_mean,
        "revealed_counts": revealed_counts,
    }
    return x, entropy_steps, meta, trace
