from __future__ import annotations

import dataclasses
from typing import Dict, Optional

from datasets import Dataset, load_dataset


@dataclasses.dataclass
class DatasetConfig:
    """Simple config describing how to read a dataset."""

    name: str
    split: str = "test"
    subset: Optional[str] = None
    question_field: str = "question"
    answer_field: str = "answer"
    metadata_fields: Optional[Dict[str, str]] = None


# Built-in presets for convenience.
DATASET_PRESETS: Dict[str, DatasetConfig] = {
    "gsm8k": DatasetConfig(name="gsm8k", split="test", question_field="question", answer_field="answer"),
    # Add other reasoning sets here.
}


def resolve_dataset_config(dataset: str, split: Optional[str] = None) -> DatasetConfig:
    """Return a DatasetConfig using presets when available."""
    if dataset in DATASET_PRESETS:
        cfg = dataclasses.replace(DATASET_PRESETS[dataset])
    else:
        cfg = DatasetConfig(name=dataset)

    if split:
        cfg.split = split
    return cfg


def load_reasoning_dataset(cfg: DatasetConfig, streaming: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """Load a datasets.Dataset according to the provided configuration."""
    data = load_dataset(
        cfg.name,
        cfg.subset,
        split=cfg.split,
        streaming=streaming,
        cache_dir=cache_dir,
    )
    return data
