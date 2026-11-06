#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from rcc_dllm.data import build_dataset
from rcc_dllm.generation import generate_with_entropy
from rcc_dllm.utils import append_jsonl, safe_name, save_entropy_trace, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLaDA entropy and reveal trajectories.")
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="triviaqa", choices=["triviaqa", "hotpotqa", "commonsenseqa"])
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--subsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    entropy_dir = output_dir / "entropy"
    manifest_path = output_dir / "manifest.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    entropy_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest_path.unlink()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
    ).to(device)
    model.eval()

    special_token_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    for attr in ("eos_token_id", "pad_token_id", "bos_token_id", "mask_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            special_token_ids.add(int(value))

    dataset = build_dataset(
        tokenizer,
        name=args.dataset,
        split=args.split,
        subsample=args.subsample,
        seed=args.seed,
        config_name=args.dataset_config,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=0)

    write_json(
        output_dir / "run_config.json",
        {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "split": args.split,
            "subsample": args.subsample,
            "diffusion_steps": args.diffusion_steps,
            "gen_length": args.gen_length,
            "temperature": args.temperature,
            "cfg_scale": args.cfg_scale,
            "mask_id": args.mask_id,
        },
    )

    global_index = 0
    for batch in tqdm(loader, desc="batches"):
        input_ids = batch["input_ids"].to(device)
        items = batch["items"]
        x, entropy_steps, meta, step_trace = generate_with_entropy(
            model,
            input_ids,
            steps=args.diffusion_steps,
            gen_length=args.gen_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            mask_id=args.mask_id,
            special_token_ids=sorted(special_token_ids),
        )
        entropy_stack = np.stack(entropy_steps, axis=0)
        decoded = tokenizer.batch_decode(x[:, input_ids.shape[1] :], skip_special_tokens=True)

        for b, item in enumerate(items):
            stem = safe_name(f"sample_{global_index:06d}_{item['question_id']}")
            entropy_path = entropy_dir / f"{stem}.npz"
            save_entropy_trace(
                entropy_path,
                entropy=entropy_stack[:, b, :],
                meta=meta,
                step_trace=step_trace,
                batch_index=b,
            )

            row = {
                "index": global_index,
                "dataset": args.dataset,
                "model_family": "llada",
                "question_id": item["question_id"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer_aliases": item.get("answer_aliases", []),
                "generation": decoded[b],
                "entropy_file": str(entropy_path.relative_to(output_dir)),
                "meta": meta,
                "hallucination": None,
            }
            if args.dataset == "commonsenseqa":
                row["choice_labels"] = item.get("choice_labels", [])
                row["choice_texts"] = item.get("choice_texts", [])
                row["answer_key"] = item.get("answer_key", "")
            append_jsonl(manifest_path, row)
            global_index += 1

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved entropy traces: {entropy_dir}")


if __name__ == "__main__":
    main()
