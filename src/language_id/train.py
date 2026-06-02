"""Train a single-language LID detector.

Turns a single-language dataset into a binary detection problem *is this
sentence the target language or not?* by pairing its sentences (positives)
with other-language sentences from Common Voice LID (negatives), then fits one
of two specialists:

- `train_logreg`: TF-IDF character n-grams + logistic regression (CPU, fast).
- `finetune_llm`: fine-tune any Hugging Face model with a sequence-
  classification head (e.g. `Qwen/Qwen3-0.6B`). Swap `model_id` to try another
  base model — nothing else changes. Needs the optional `finetune` extra.
"""


from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from language_id.data import (
    LANG_COLUMN_NAME,
    TEXT_COLUMN_NAME,
    load_commonvoice_lid,
    load_single_language_text_archive,
)
from language_id.lang_codes_mapping import to_iso3
from language_id.models.base import LIDModel, LIDPrediction

# Label used for every non-target sentence in the binary problem.
OTHER_LABEL = "other"

DEFAULT_HF_MODEL_ID = "Qwen/Qwen3-0.6B"


class CharNgramLID:
    """A trained char n-gram + logistic-regression pipeline as a LID model.

    Predicts the target ISO-639-3 code or `OTHER_LABEL`, with the classifier's
    max class probability as confidence.
    """

    def __init__(self, pipeline: Pipeline, name: str = "charngram-lr"):
        self.pipeline = pipeline
        self.name = name

    def predict(self, text: str) -> LIDPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        labels = self.pipeline.predict(texts)
        conf = self.pipeline.predict_proba(texts).max(axis=1)
        return [
            LIDPrediction(lang_code=str(label), confidence=float(c), raw_output=str(label))
            for label, c in zip(labels, conf, strict=True)
        ]


class HFTransformerLID:
    """A fine-tuned HF sequence-classification model as a LID model.

    Predicts the target ISO-639-3 code or `OTHER_LABEL`, with the softmax
    probability of the chosen class as confidence.
    """

    def __init__(self, model, tokenizer, id2label, name, max_len: int = 256, batch_size: int = 32):
        import torch

        self._torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.name = name
        self.max_len = max_len
        self.batch_size = batch_size

    def predict(self, text: str) -> LIDPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        torch = self._torch
        out: list[LIDPrediction] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [str(t) for t in texts[start : start + self.batch_size]]
            enc = self.tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=self.max_len, padding=True
            ).to(self.model.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)
            conf, idx = probs.max(dim=-1)
            for c, j in zip(conf.tolist(), idx.tolist(), strict=True):
                label = self.id2label[int(j)]
                out.append(
                    LIDPrediction(lang_code=str(label), confidence=float(c), raw_output=str(label))
                )
        return out


def build_training_data(
    target_dataset: str,
    target_lang: str,
    *,
    n_train: int | None = None,
    n_neg: int | None = None,
    seed: int = 0,
    test_size: float = 0.2,
    download_directory: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble binary train/test frames for a single-language detector.

    Positives are the target dataset's sentences (loaded from its `.tar.gz`
    archive); negatives are other-language sentences sampled from Common Voice
    LID. Negatives mirror the positives 1:1 in each split, so the classes stay
    balanced.

    Args:
        n_train: cap on the number of target-language *training* samples (the
            negatives match it). `None` uses every positive training row. The
            test set is never capped, so models stay comparable across sizes.
        n_neg: override the number of training negatives (defaults to the
            positive training count).

    Returns `(train_df, test_df)`:
    - `train_df`: columns `[sentence, label]`, where `label` is the target
      ISO-639-3 code or `OTHER_LABEL` (shuffled).
    - `test_df`: columns `[sentence, lang, is_target]`, keeping each row's true
      language so every model can be asked the same yes/no question.
    """
    target_lang = to_iso3(target_lang)

    pos = load_single_language_text_archive(
        target_dataset, target_lang, download_directory=download_directory
    )
    # The corpus has no split column, so create a train/test split ourselves.
    pos_train, pos_test = train_test_split(pos, test_size=test_size, random_state=seed)
    # Optionally cap the number of target-language training samples. The test
    # set is left untouched so results stay comparable across training sizes.
    if n_train is not None and n_train < len(pos_train):
        pos_train = pos_train.sample(n=n_train, random_state=seed)

    cv = load_commonvoice_lid()
    cv = cv[cv[LANG_COLUMN_NAME] != target_lang]  # never use the target language as a negative
    cv = cv.dropna(subset=[TEXT_COLUMN_NAME])
    # Mirror the positives: as many training negatives as positive training
    # samples (overridable via n_neg), plus enough to balance the fixed test set.
    n_neg_train = len(pos_train) if n_neg is None else n_neg
    n_neg_test = len(pos_test)
    neg = cv.sample(n=min(n_neg_train + n_neg_test, len(cv)), random_state=seed)[
        [TEXT_COLUMN_NAME, LANG_COLUMN_NAME]
    ].copy()
    neg[TEXT_COLUMN_NAME] = neg[TEXT_COLUMN_NAME].map(str)
    neg_test = neg.iloc[:n_neg_test]
    neg_train = neg.iloc[n_neg_test : n_neg_test + n_neg_train]

    train_df = (
        pd.concat(
            [
                pos_train[[TEXT_COLUMN_NAME]].assign(label=target_lang),
                neg_train[[TEXT_COLUMN_NAME]].assign(label=OTHER_LABEL),
            ],
            ignore_index=True,
        )
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    test_df = pd.concat(
        [
            pos_test[[TEXT_COLUMN_NAME, LANG_COLUMN_NAME]],
            neg_test[[TEXT_COLUMN_NAME, LANG_COLUMN_NAME]],
        ],
        ignore_index=True,
    )
    test_df["is_target"] = test_df[LANG_COLUMN_NAME] == target_lang
    return train_df, test_df


def train_logreg(
    train_df: pd.DataFrame,
    *,
    analyzer: str = "char_wb",
    ngram_range: tuple[int, int] = (2, 5),
    min_df: int = 2,
    max_iter: int = 1000,
    name: str = "charngram-lr",
) -> CharNgramLID:
    """Fit a TF-IDF char n-gram + logistic-regression detector on `train_df`.

    `train_df` must have `sentence` and `label` columns (see
    `build_training_data`). `class_weight='balanced'` compensates for class
    imbalance when there are more negatives than positives.
    """
    clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df)),
            ("lr", LogisticRegression(max_iter=max_iter, class_weight="balanced")),
        ]
    )
    clf.fit(train_df[TEXT_COLUMN_NAME], train_df["label"])
    return CharNgramLID(clf, name=name)


def finetune_llm(
    train_df: pd.DataFrame,
    target_lang: str,
    *,
    model_id: str = DEFAULT_HF_MODEL_ID,
    epochs: float = 3,
    batch_size: int = 8,
    max_len: int = 256,
    learning_rate: float = 2e-5,
    output_dir: str | Path | None = None,
    seed: int = 0,
    name: str | None = None,
    show_progress: bool = True,
) -> HFTransformerLID:
    """Fine-tune a HF model with a classification head as a binary detector.

    `model_id` is any Hugging Face model id with (or compatible with) a
    sequence-classification head — most causal LLMs work via
    `AutoModelForSequenceClassification`. Swap it to try a different base model;
    nothing else needs to change.

    Requires the optional `finetune` extra (torch / transformers / datasets /
    accelerate); install it with `uv sync --extra finetune`.

    `show_progress` prints percentage complete, elapsed time, and an ETA roughly
    every 5% of training steps (in addition to the default tqdm bar), so a long
    fine-tune reports how much is left.
    """
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as e:
        raise ImportError(
            "LLM fine-tuning needs the optional 'finetune' extra "
            "(torch/transformers/datasets/accelerate). Install it with: "
            "uv sync --extra finetune"
        ) from e

    target_lang = to_iso3(target_lang)
    label2id = {OTHER_LABEL: 0, target_lang: 1}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_pandas(
        train_df.assign(labels=train_df["label"].map(label2id))[[TEXT_COLUMN_NAME, "labels"]],
        preserve_index=False,
    ).map(
        lambda batch: tokenizer(batch[TEXT_COLUMN_NAME], truncation=True, max_length=max_len),
        batched=True,
        remove_columns=[TEXT_COLUMN_NAME],
    )

    # Load in float32 so fp16 mixed-precision training works on any GPU (some
    # checkpoints ship bf16 weights, which the fp16 grad scaler can't unscale).
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, label2id=label2id, id2label=id2label, dtype=torch.float32
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    output_dir = output_dir or (Path("hf-lid-out") / model_id.replace("/", "_"))
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    class _ProgressCallback(TrainerCallback):
        """Print %-complete, elapsed, and ETA roughly every `every_frac` of steps."""

        def __init__(self, every_frac: float = 0.05):
            self.every_frac = every_frac
            self._next = every_frac
            self._start = 0.0

        def on_train_begin(self, args, state, control, **kwargs):
            import time

            self._start = time.time()
            self._next = self.every_frac
            steps_per_epoch = state.max_steps // max(1, round(args.num_train_epochs))
            print(
                f"[finetune] {model_id}: {state.max_steps} steps "
                f"({args.num_train_epochs:g} epochs x ~{steps_per_epoch} steps, "
                f"batch size {args.per_device_train_batch_size}) on "
                f"{'GPU' if torch.cuda.is_available() else 'CPU'}",
                flush=True,
            )

        def on_step_end(self, args, state, control, **kwargs):
            import time

            if not state.max_steps:
                return
            frac = state.global_step / state.max_steps
            if frac + 1e-9 < self._next and state.global_step != state.max_steps:
                return
            self._next = frac + self.every_frac
            elapsed = time.time() - self._start
            eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
            print(
                f"[finetune] {frac:4.0%}  step {state.global_step}/{state.max_steps}  "
                f"elapsed {elapsed / 60:5.1f}m  ETA {eta / 60:5.1f}m",
                flush=True,
            )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        callbacks=[_ProgressCallback()] if show_progress else None,
    )
    trainer.train()

    name = name or f"ft-{model_id.split('/')[-1]}"
    return HFTransformerLID(model, tokenizer, id2label, name=name, max_len=max_len)


def evaluate_detector(model: LIDModel, test_df: pd.DataFrame, target_lang: str) -> dict[str, Any]:
    """Score a trained detector on `test_df`'s binary `is_target` gold.

    Reduces every prediction to yes/no — *is it the target language?* — and
    returns precision/recall/F1/accuracy for that detection task.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    target_lang = to_iso3(target_lang)
    preds = model.predict_batch(test_df[TEXT_COLUMN_NAME].tolist())
    y_pred = [to_iso3(p.lang_code) == target_lang for p in preds]
    y_true = test_df["is_target"].tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "model": getattr(model, "name", type(model).__name__),
        "n": len(test_df),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy_score(y_true, y_pred),
    }
