from language_id.data import load_commonlid, sample

df = load_commonlid()
# df = load_commonvoicelid()
df_subset = sample(df, n_per_lang=10, langs=["eng", "fra", "deu"], seed=42, min_sentence_length=)
