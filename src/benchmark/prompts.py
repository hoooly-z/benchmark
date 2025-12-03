from __future__ import annotations

from textwrap import dedent
from typing import Dict

DEFAULT_PROMPT = dedent(
    """
    You are a meticulous math tutor. Solve the following problem step by step and
    clearly mark the final answer on a new line starting with "Answer:".

    Question:
    {question}
    """
).strip()


def render_prompt(question: str, template: str = DEFAULT_PROMPT, context: Dict[str, str] | None = None) -> str:
    """Render a prompt safely even when context keys are missing."""
    ctx = {"question": question}
    if context:
        ctx.update(context)
    return template.format(**ctx)
