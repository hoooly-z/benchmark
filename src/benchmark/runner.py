from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .data import DatasetConfig
from .eval import compare_answers
from .prompts import DEFAULT_PROMPT, render_prompt


def prepare_tokenizer(model_name: str, cache_dir: Optional[str] = None) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def prepare_model(model_name: str, device: str = "auto", cache_dir: Optional[str] = None) -> PreTrainedModel:
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map: str | int | None
    if device == "auto":
        device_map = "auto" if torch.cuda.is_available() else None
    elif device == "cpu":
        device_map = "cpu"
    else:
        device_map = device

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    return model


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_samples(dataset: Dataset, max_samples: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    count = 0
    for row in dataset:
        yield row
        count += 1
        if max_samples and count >= max_samples:
            break


def build_prompt(sample: Dict[str, Any], question_field: str, template: str) -> str:
    question = sample[question_field]
    return render_prompt(question=question, template=template, context={k: str(v) for k, v in sample.items()})


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    gen_cfg: GenerationConfig,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**encoded, generation_config=gen_cfg)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove the prompt portion so that we only keep the completion.
    completion = text[len(prompt) :].strip()
    return completion or text


def run_model_on_dataset(
    cfg: DatasetConfig,
    dataset: Dataset,
    model_name: str,
    output_path: Path,
    prompt_template: str | None = None,
    max_samples: Optional[int] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    device: str = "auto",
    evaluate: bool = True,
) -> None:
    ensure_output_dir(output_path)
    template = prompt_template or DEFAULT_PROMPT
    tokenizer = prepare_tokenizer(model_name, cache_dir=cache_dir)
    model = prepare_model(model_name, device=device, cache_dir=cache_dir)

    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        **(generation_kwargs or {}),
    )

    question_field = cfg.question_field
    answer_field = cfg.answer_field

    total_evaluated = 0
    total_correct = 0

    with output_path.open("w", encoding="utf-8") as f:
        samples = iter_samples(dataset, max_samples)
        for idx, sample in enumerate(tqdm(samples, desc="Running benchmark")):
            prompt = build_prompt(sample, question_field=question_field, template=template)
            start = time.perf_counter()
            generation = generate_text(model=model, tokenizer=tokenizer, prompt=prompt, gen_cfg=gen_config)
            elapsed = time.perf_counter() - start

            reference = sample.get(answer_field)
            is_correct = None
            if evaluate and reference:
                comparison = compare_answers(generation, reference)
                is_correct = comparison
                if comparison is not None:
                    total_evaluated += 1
                    if comparison:
                        total_correct += 1

            record = {
                "id": sample.get("id", f"{cfg.name}-{cfg.split}-{idx}"),
                "question": sample.get(question_field),
                "reference_answer": reference,
                "prompt": prompt,
                "generation": generation,
                "completion_time_sec": elapsed,
                "model_name": model_name,
                "dataset": cfg.name,
                "dataset_config": asdict(cfg),
                "is_correct": is_correct,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    if evaluate and total_evaluated:
        accuracy = total_correct / total_evaluated
        print(f"[benchmark] Accuracy: {total_correct}/{total_evaluated} = {accuracy:.2%}")
