from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .data import load_reasoning_dataset, resolve_dataset_config
from .runner import run_model_on_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reasoning benchmark and save generations.")
    parser.add_argument("--model-name", required=True, help="HF repo id or local path.")
    parser.add_argument("--dataset", default="gsm8k", help="Dataset preset or HF dataset name.")
    parser.add_argument("--split", default=None, help="Dataset split (overrides preset).")
    parser.add_argument("--subset", default=None, help="Dataset subset/config name.")
    parser.add_argument("--question-field", default=None, help="Field containing the question.")
    parser.add_argument("--answer-field", default=None, help="Field containing the answer.")
    parser.add_argument("--output", required=True, help="Path to JSONL output.")
    parser.add_argument("--prompt-template", default=None, help="Custom template with {question} placeholder.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for smoke tests.")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache location.")
    parser.add_argument("--device", default="auto", help="Device spec passed to transformers (auto/cpu/cuda).")
    parser.add_argument(
        "--generation-config",
        default=None,
        help="Optional JSON dict of GenerationConfig overrides.",
    )
    parser.add_argument("--streaming", action="store_true", help="Enable streaming dataset loading.")
    parser.add_argument("--disable-eval", action="store_true", help="Skip answer-matching accuracy computation.")
    return parser


def parse_generation_kwargs(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    import json

    return json.loads(raw)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = resolve_dataset_config(args.dataset, split=args.split)
    if args.subset:
        cfg.subset = args.subset
    if args.question_field:
        cfg.question_field = args.question_field
    if args.answer_field:
        cfg.answer_field = args.answer_field

    dataset = load_reasoning_dataset(cfg, streaming=args.streaming, cache_dir=args.cache_dir)
    prompt_template = args.prompt_template

    run_model_on_dataset(
        cfg=cfg,
        dataset=dataset,
        model_name=args.model_name,
        output_path=Path(args.output),
        prompt_template=prompt_template or None,
        max_samples=args.max_samples,
        generation_kwargs=parse_generation_kwargs(args.generation_config),
        cache_dir=args.cache_dir,
        device=args.device,
        evaluate=not args.disable_eval,
    )


if __name__ == "__main__":
    main()
