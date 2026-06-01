# language-id

Benchmarking and building text language identification with Common Crawl and Mozilla Data Collective.

A reproducible benchmarking project comparing frontier LLMs, standard tools, and custom-trained classifiers on text-based language identification across the web and transcribed-speech domains. Built around two open benchmarks — **CommonLID** (Common Crawl Foundation) and **CommonVoiceLID** (Mozilla Data Collective) — with explicit support for adding new languages.

See [`spec.md`](spec.md) for the full specification.

## Quickstart

```bash
uv sync
uv run language-id --help
```

## Layout

- `src/language_id/` — package source
- `configs/` — model, experiment, and resource-tier YAML
- `docs/` — mkdocs site, deployed to GH Pages
- `results/` — local run artifacts (gitignored)

## Status

Scaffolding only. See `spec.md` §18 for the full implementation order.
