from __future__ import annotations

from textwrap import dedent
from typing import Dict

DEFAULT_PROMPT = dedent(
    """
    You are a meticulous math tutor. Solve the problem step by step.
    At the end, output the final numeric answer in the format:

    #### <number>

    Question:
    {question}

    Answer:
    """
).strip()


def render_prompt(question: str, template: str = DEFAULT_PROMPT, context: Dict[str, str] | None = None) -> str:
    """Render a prompt safely even when context keys are missing."""
    ctx = {"question": question}
    if context:
        ctx.update(context)
    return template.format(**ctx)
