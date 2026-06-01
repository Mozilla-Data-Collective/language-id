# Experiment 1: Off-the-shelf evaluation on CommonLID

Benchmark frontier LLMs and standard baselines on a length-stratified subsample of CommonLID.

## Models

**Frontier LLMs** (exact snapshot IDs locked in `configs/models/*.yaml` at experiment start):

- GPT-5.X (OpenAI)
- Claude Opus 4.X (Anthropic)
- Qwen 3.X (32B)
- Mistral Large 2

**Standard / specialist baselines**:

- `langdetect`
- GlotLID
- NLLB-200 LID head

## Output normalization

All model outputs are normalized to BCP-47. Unparseable LLM responses are tracked as a first-class metric.

## Running

```bash
uv run language-id eval --model gpt-5 --dataset commonlid --config configs/experiments/exp1_offshelf_eval.yaml
```

Or run the orchestrator directly:

```bash
uv run python -m language_id.experiments.exp1_offshelf_eval configs/experiments/exp1_offshelf_eval.yaml
```
