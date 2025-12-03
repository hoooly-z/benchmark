from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

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


# ===========================
# Tokenizer
# ===========================
def prepare_tokenizer(model_name: str, cache_dir: Optional[str] = None) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ===========================
# Model (可用 bf16；flash-attn2 自动关闭以避免报错)
# ===========================
def prepare_model(model_name: str, device: str = "auto", cache_dir: Optional[str] = None) -> PreTrainedModel:
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        # 如果以后安装了 flash-attn，可以改为：attn_implementation="flash_attention_2",
        attn_implementation=None,  # 保证现在环境不会报错
        cache_dir=cache_dir,
    )
    return model


# ===========================
# 样本迭代器
# ===========================
def iter_samples(dataset: Dataset, max_samples: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    count = 0
    for row in dataset:
        yield row
        count += 1
        if max_samples and count >= max_samples:
            break


# ===========================
# Prompt Builder
# ===========================
def build_prompt(sample: Dict[str, Any], question_field: str, template: str) -> str:
    question = sample[question_field]
    return render_prompt(
        question=question,
        template=template,
        context={k: str(v) for k, v in sample.items()}
    )


# ===========================
# 批量生成（核心加速）
# ===========================
def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    gen_cfg: GenerationConfig,
) -> List[str]:

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            generation_config=gen_cfg,
        )

    texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # 去掉 prompt 部分，只保留 completion
    completions = [
        texts[i][len(prompts[i]):].strip()
        for i in range(len(prompts))
    ]

    return completions


# ===========================
# 保存一批结果
# ===========================
def save_batch_results(
    f,
    cfg,
    model_name,
    prompts,
    samples,
    completions,
    answer_field,
    elapsed,
    total_evaluated,
    total_correct,
    evaluate=True,
):
    per_sample_time = elapsed / len(prompts)

    for prompt, sample, completion in zip(prompts, samples, completions):
        reference = sample.get(answer_field)
        is_correct = None

        if evaluate and reference:
            comp = compare_answers(completion, reference)
            is_correct = comp
            if comp is not None:
                total_evaluated += 1
                if comp:
                    total_correct += 1

        record = {
            "id": sample.get("id"),
            "question": sample.get(cfg.question_field),
            "reference_answer": reference,
            "prompt": prompt,
            "generation": completion,
            "completion_time_sec": per_sample_time,
            "model_name": model_name,
            "dataset": cfg.name,
            "dataset_config": asdict(cfg),
            "is_correct": is_correct,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

    return total_evaluated, total_correct


# ===========================
# 主执行逻辑（已优化）
# ===========================
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
    batch_size: int = 16,        # ⭐ 批量默认 16（3090 非常合适）
    evaluate: bool = True,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    template = prompt_template or DEFAULT_PROMPT

    tokenizer = prepare_tokenizer(model_name, cache_dir=cache_dir)
    model = prepare_model(model_name, device=device, cache_dir=cache_dir)

    gen_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        **(generation_kwargs or {}),
    )

    question_field = cfg.question_field
    answer_field = cfg.answer_field

    total_evaluated = 0
    total_correct = 0

    # tqdm total 修复
    total_len = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    samples = iter_samples(dataset, max_samples)

    pending_prompts: List[str] = []
    pending_samples: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(
            tqdm(samples, total=total_len, desc="Running benchmark", unit="sample")
        ):
            prompt = build_prompt(sample, question_field=question_field, template=template)

            pending_prompts.append(prompt)
            pending_samples.append(sample)

            # 批量处理
            if len(pending_prompts) >= batch_size:
                start = time.perf_counter()
                completions = batch_generate(model, tokenizer, pending_prompts, gen_config)
                elapsed = time.perf_counter() - start

                total_evaluated, total_correct = save_batch_results(
                    f,
                    cfg,
                    model_name,
                    pending_prompts,
                    pending_samples,
                    completions,
                    answer_field,
                    elapsed,
                    total_evaluated,
                    total_correct,
                    evaluate,
                )

                pending_prompts = []
                pending_samples = []

        # 处理剩余不足 batch 的
        if pending_prompts:
            start = time.perf_counter()
            completions = batch_generate(model, tokenizer, pending_prompts, gen_config)
            elapsed = time.perf_counter() - start

            total_evaluated, total_correct = save_batch_results(
                f,
                cfg,
                model_name,
                pending_prompts,
                pending_samples,
                completions,
                answer_field,
                elapsed,
                total_evaluated,
                total_correct,
                evaluate,
            )

    if evaluate and total_evaluated:
        accuracy = total_correct / total_evaluated
        print(f"[benchmark] Accuracy: {total_correct}/{total_evaluated} = {accuracy:.2%}")

