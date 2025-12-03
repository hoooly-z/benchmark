# Math Reasoning Benchmark Harness

This repo provides a minimal harness for running open‑source LLMs on math reasoning datasets
such as GSM8K and storing the raw generations for downstream evaluation.

## Features

- Pulls datasets via `datasets.load_dataset` with dataset-specific field mappings.
- Runs any Hugging Face causal language model (AutoModelForCausalLM) with configurable
  decoding hyper‑parameters.
- Streams JSONL logs capturing prompts, generations, references, and timing metadata.
- Built-in extractor matches GSM8K-style `#### answer` targets so you immediately get
  per-sample correctness plus aggregate accuracy stats.
- Includes a light prompt templating utility with sensible defaults for GSM8K style tasks.

## Quickstart

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run a benchmark (example uses a small model for smoke‑testing):

   ```bash
   python -m src.benchmark.run_benchmark ^
     --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
     --dataset gsm8k ^
     --split test ^
     --output runs/tinyllama_gsm8k.jsonl ^
     --max-samples 32
   ```

   Replace the model with whatever local/remote checkpoint you have access to.
   When using larger models ensure you configure `--device cuda` (default is auto).

## CLI Overview

`python -m src.benchmark.run_benchmark --help`

- `--model-name`: Hugging Face repo or local path.
- `--dataset`: Currently `gsm8k` (default template) or any HF dataset name when
  combined with `--question-field/--answer-field`.
- `--split`: Dataset split to run on.
- `--max-samples`: Optional limit for debugging.
- `--prompt-template`: Custom template using `{question}` and optional `{context}` placeholders.
- `--output`: JSONL destination; directories are created automatically.
- `--disable-eval`: Skip the built-in GSM8K answer matcher when you only need raw generations.

Each row in the output file looks like:

```json
{
  "id": "gsm8k-train-42",
  "question": "...",
  "reference_answer": "###",
  "prompt": "system+user prompt text",
  "generation": "model response",
  "completion_time_sec": 2.31,
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "dataset": "gsm8k",
  "is_correct": true
}
```

## Extending

- Add more dataset presets in `src/benchmark/data.py`.
- Customize prompt templates or add few‑shot exemplars in `src/benchmark/prompts.py`.
- Implement secondary evaluation scripts that parse the saved generations and score
  them with matching or judge models.

## Notes

- Downloading datasets and checkpoints requires Hugging Face credentials if they are gated.
- This harness only captures generations; evaluation logic is intentionally separate so
  you can experiment with rule-based or LLM-as-a-judge methods downstream.
