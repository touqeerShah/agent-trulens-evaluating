"""Prompt files are intentionally kept separate so learners can edit them."""

from pathlib import Path

PROMPT_DIR = Path(__file__).with_name("prompt_files")


def load_prompt(name: str, **values: object) -> str:
    template = (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return template.format(**values)
