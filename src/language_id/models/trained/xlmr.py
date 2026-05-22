"""XLM-RoBERTa-large fine-tuned with LoRA via HF Trainer (spec §3 Experiment 2, §5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from language_id.models.base import LIDPrediction

_PREDICT_BATCH_SIZE = 32


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


class XLMRModel:
    """XLM-RoBERTa-large fine-tuned with LoRA (or optionally full) for LID."""

    name = "xlmr-lora"
    version = "0.0.1"

    def __init__(
        self,
        base_model: str = "xlm-roberta-large",
        max_length: int = 256,
        finetune: dict[str, Any] | None = None,
        training: dict[str, Any] | None = None,
    ) -> None:
        self.base_model = base_model
        self.max_length = max_length
        self.finetune = dict(finetune or {})
        self.training = dict(training or {})
        self.model: Any = None
        self.tokenizer: Any = None
        self.label2id: dict[str, int] | None = None
        self.id2label: dict[int, str] | None = None

    def _build_label_index(self, langs: list[str]) -> None:
        labels = sorted(set(langs))
        self.label2id = {lang: i for i, lang in enumerate(labels)}
        self.id2label = {i: lang for lang, i in self.label2id.items()}

    def _tokenize_df(self, df: pd.DataFrame, text_col: str, lang_col: str) -> Dataset:
        assert self.tokenizer is not None and self.label2id is not None
        ds = Dataset.from_pandas(df[[text_col, lang_col]].reset_index(drop=True))
        tokenizer = self.tokenizer
        label2id = self.label2id
        max_length = self.max_length

        def encode(batch: dict[str, list[Any]]) -> dict[str, Any]:
            enc = tokenizer(batch[text_col], truncation=True, max_length=max_length)
            enc["labels"] = [label2id[lang] for lang in batch[lang_col]]
            return enc

        return ds.map(encode, batched=True, remove_columns=ds.column_names)

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        text_col: str,
        lang_col: str,
        output_dir: Path | str = "results/models/xlmr-lora-tmp",
    ) -> dict[str, float]:
        all_langs = list(train_df[lang_col]) + list(eval_df[lang_col])
        self._build_label_index(all_langs)
        n_labels = len(self.label2id or {})

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        base = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=n_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        strategy = self.finetune.get("strategy", "lora")
        if strategy == "lora":
            lora_cfg_dict = self.finetune.get("lora", {})
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=int(lora_cfg_dict.get("r", 16)),
                lora_alpha=int(lora_cfg_dict.get("alpha", 32)),
                lora_dropout=float(lora_cfg_dict.get("dropout", 0.05)),
                target_modules=lora_cfg_dict.get(
                    "target_modules", ["query", "key", "value"]
                ),
            )
            self.model = get_peft_model(base, lora_cfg)
        else:
            self.model = base

        train_ds = self._tokenize_df(train_df, text_col, lang_col)
        eval_ds = self._tokenize_df(eval_df, text_col, lang_col)

        args = TrainingArguments(output_dir=str(output_dir), **self.training)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=_compute_metrics,
        )
        trainer.train()
        self.model = trainer.model
        return {k: float(v) for k, v in trainer.evaluate().items() if isinstance(v, (int, float))}

    def predict(self, text: str) -> LIDPrediction:
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        if self.model is None or self.tokenizer is None or self.id2label is None:
            raise RuntimeError("XLMRModel not fit yet; call fit() or load()")
        self.model.eval()
        device = next(self.model.parameters()).device
        results: list[LIDPrediction] = []
        for i in range(0, len(texts), _PREDICT_BATCH_SIZE):
            batch = texts[i : i + _PREDICT_BATCH_SIZE]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in probs:
                idx = int(np.argmax(p))
                results.append(
                    LIDPrediction(
                        lang_code=self.id2label[idx],
                        confidence=float(p[idx]),
                        raw_output="",
                    )
                )
        return results

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        if self.model is None or self.tokenizer is None or self.id2label is None:
            raise RuntimeError("XLMRModel not fit yet; call fit() or load()")
        self.model.eval()
        device = next(self.model.parameters()).device
        out: list[dict[str, float]] = []
        for i in range(0, len(texts), _PREDICT_BATCH_SIZE):
            batch = texts[i : i + _PREDICT_BATCH_SIZE]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in probs:
                out.append({self.id2label[i]: float(p[i]) for i in range(len(p))})
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("XLMRModel not fit yet; nothing to save")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "labels.json").write_text(
            json.dumps(
                {
                    "label2id": self.label2id,
                    "id2label": {str(k): v for k, v in (self.id2label or {}).items()},
                }
            )
        )
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "version": self.version,
                    "base_model": self.base_model,
                    "max_length": self.max_length,
                    "finetune": self.finetune,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "XLMRModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        labels_data = json.loads((path / "labels.json").read_text())
        instance = cls(
            base_model=meta.get("base_model", "xlm-roberta-large"),
            max_length=int(meta.get("max_length", 256)),
            finetune=meta.get("finetune") or {},
        )
        instance.label2id = labels_data["label2id"]
        instance.id2label = {int(k): v for k, v in labels_data["id2label"].items()}
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        if (path / "adapter_config.json").exists():
            peft_cfg = PeftConfig.from_pretrained(path)
            base = AutoModelForSequenceClassification.from_pretrained(
                peft_cfg.base_model_name_or_path,
                num_labels=len(instance.label2id),
                id2label=instance.id2label,
                label2id=instance.label2id,
            )
            instance.model = PeftModel.from_pretrained(base, path)
        else:
            instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        return instance
