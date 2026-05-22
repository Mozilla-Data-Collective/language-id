# Experiment 2 — Train three classifiers on CommonVoiceLID

Three complementary paradigms:

1. **Logistic Regression** on character n-grams — sklearn `Pipeline`: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))` → `LogisticRegression`.
2. **Multinomial Naive Bayes** on character n-grams — same vectorizer, `MultinomialNB`.
3. **XLM-RoBERTa-large** fine-tuned with **LoRA** (rank 16, attention projection layers) via HuggingFace `Trainer`.

Hyperparameter tuning happens on a held-out slice of CommonVoiceLID `dev`. Splits are used as-is (do not redefine).

## Running

```bash
uv run language-id train --model logreg   --dataset commonvoice_lid --config configs/experiments/exp2_train_classifiers.yaml
uv run language-id train --model ngram_nb --dataset commonvoice_lid --config configs/experiments/exp2_train_classifiers.yaml
uv run language-id train --model xlmr     --dataset commonvoice_lid --config configs/experiments/exp2_train_classifiers.yaml
```
