from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset


SYSTEM_PROMPT = "You are a helpful assistant. Answer the question concisely and factually."


@dataclass
class DatasetConfig:
    name: str
    split: str = "validation"
    subsample: int = 100
    seed: int = 42
    system_prompt: str = SYSTEM_PROMPT
    config_name: str | None = None


class QADataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, cfg: DatasetConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.dataset = self._load_dataset()
        n = len(self.dataset)
        if cfg.subsample and cfg.subsample > 0:
            rng = np.random.default_rng(cfg.seed)
            self.indices = rng.choice(n, size=min(cfg.subsample, n), replace=False)
        else:
            self.indices = np.arange(n)

    def _load_dataset(self):
        name = self.cfg.name
        if name == "triviaqa":
            return load_dataset("trivia_qa", self.cfg.config_name or "rc.nocontext", split=self.cfg.split)
        if name == "hotpotqa":
            return load_dataset("hotpot_qa", self.cfg.config_name or "distractor", split=self.cfg.split)
        if name == "commonsenseqa":
            return load_dataset("commonsense_qa", split=self.cfg.split)
        raise ValueError(f"Unsupported dataset: {name}")

    def __len__(self) -> int:
        return len(self.indices)

    def _chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataset[int(self.indices[idx])]
        name = self.cfg.name

        if name == "triviaqa":
            question = str(row["question"]).strip()
            answer_obj = row.get("answer", {})
            value = str(answer_obj.get("value", "")).strip()
            aliases = [str(x).strip() for x in answer_obj.get("aliases", []) if str(x).strip()]
            user_prompt = f"Answer the question directly. Return only the answer, without explanation.\n\nQuestion: {question}\n\nAnswer:"
            return {
                "question_id": row.get("question_id", str(idx)),
                "question": question,
                "ground_truth": value,
                "answer_aliases": aliases or ([value] if value else []),
                "prompt_text": self._chat_prompt(user_prompt),
            }

        if name == "hotpotqa":
            question = str(row["question"]).strip()
            answer = str(row.get("answer", "")).strip()
            user_prompt = f"Answer the question directly. Return only the answer, without explanation.\n\nQuestion: {question}\n\nAnswer:"
            return {
                "question_id": row.get("id", str(idx)),
                "question": question,
                "ground_truth": answer,
                "answer_aliases": [answer] if answer else [],
                "prompt_text": self._chat_prompt(user_prompt),
            }

        if name == "commonsenseqa":
            question = str(row["question"]).strip()
            choices = row["choices"]
            labels = [str(x).strip() for x in choices["label"]]
            texts = [str(x).strip() for x in choices["text"]]
            answer_key = str(row.get("answerKey", "")).strip()
            answer_text = dict(zip(labels, texts)).get(answer_key, "")
            choices_text = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
            user_prompt = (
                "Answer the multiple-choice question directly. Choose the best option and return only the option label.\n\n"
                f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
            )
            return {
                "question_id": row.get("id", str(idx)),
                "question": question,
                "ground_truth": answer_text,
                "answer_aliases": [answer_key, answer_text] if answer_key else [answer_text],
                "prompt_text": self._chat_prompt(user_prompt),
                "choice_labels": labels,
                "choice_texts": texts,
                "answer_key": answer_key,
            }

        raise ValueError(name)

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [item["prompt_text"] for item in batch]
        enc = self.tokenizer(prompts, return_tensors="pt", padding="longest")
        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.get("attention_mask"),
            "items": batch,
        }


def build_dataset(tokenizer, *, name: str, split: str, subsample: int, seed: int, config_name: str | None = None) -> QADataset:
    cfg = DatasetConfig(
        name=name,
        split=split,
        subsample=subsample,
        seed=seed,
        config_name=config_name,
    )
    return QADataset(tokenizer, cfg)
