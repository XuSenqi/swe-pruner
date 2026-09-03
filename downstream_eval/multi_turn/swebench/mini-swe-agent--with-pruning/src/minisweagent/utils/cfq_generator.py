from __future__ import annotations

import re
from typing import Any

import requests
from pydantic import BaseModel, Field

from minisweagent.utils.log import logger

CFQ_SYSTEM_PROMPT = """You generate context focus questions (CFQ) for pruning large shell command outputs.

Rules:
- Output ONE complete self-contained question for the pruner
- Do NOT mention filenames or line numbers in the question
- Use the agent's reasoning to infer what it wants from the command output
- When current-step reasoning is brief (e.g. "continue looking", "read more"), use prior-step reasoning to recover the full intent
- The CFQ should capture what the agent is trying to learn from THIS command's output, not restate the command itself
- Output exactly SKIP only for: sed -i edits, git commands, python/pytest runs, ls/pwd/wc, or submission commands
- For cat/nl/grep/sed -n/head/tail reading code or search output, ALWAYS output a question

Output ONLY the question or SKIP."""

CFQ_USER_TEMPLATE = """{reasoning_block}

Command:
{command}"""

CFQ_REASONING_ONLY = """Agent reasoning:
{reasoning}"""

CFQ_REASONING_WITH_PRIOR = """Prior agent reasoning (recent steps):
{prior}

Current step reasoning:
{current}"""


class CFQGeneratorConfig(BaseModel):
    url: str = "http://10.10.10.181:8000/v1/chat/completions"
    model: str = "Qwen3.5-35B-A3B-FP8"
    timeout: float = 60.0
    retries: int = 2
    max_tokens: int = 256
    temperature: float = 0.0
    refine_existing: bool = False
    min_output_chars: int = 3200
    min_reasoning_chars: int = 150
    max_prior_reasoning_steps: int = 4
    headers: dict[str, str] = Field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=lambda: {"enable_thinking": False})

    class Config:
        extra = "allow"


class CFQGenerator:
    def __init__(self, config: CFQGeneratorConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"} | config.headers)

    def should_generate(self, command: str, output: str, *, min_output_chars: int) -> bool:
        cmd = command.strip()
        if len(output) < min_output_chars:
            return False
        skip_patterns = (
            "sed -i",
            "git diff",
            "git add",
            "git commit",
            "COMPLETE_TASK",
            "MINI_SWE_AGENT_FINAL_OUTPUT",
            "echo COMPLETE",
        )
        if any(pattern in cmd for pattern in skip_patterns):
            return False
        if cmd.startswith("python ") or cmd.startswith("python3 ") or "pytest" in cmd:
            return False
        if re.fullmatch(r"(ls|pwd|wc|echo|mkdir|rm|cp|mv)(\s+.*)?", cmd.split("&&")[0].strip()):
            return False
        read_patterns = ("cat ", "nl -ba", "grep ", "find ", "sed -n", "head ", "tail ")
        return any(pattern in cmd for pattern in read_patterns)

    @staticmethod
    def build_reasoning_block(
        current: str,
        prior: list[str] | None = None,
        *,
        min_chars: int = 150,
    ) -> tuple[str, bool]:
        """Return (reasoning_block, used_prior_context)."""
        current = current.strip()
        prior = [item.strip() for item in (prior or []) if item.strip()]
        if len(current) >= min_chars or not prior:
            return CFQ_REASONING_ONLY.format(reasoning=current), False
        prior_lines = "\n".join(f"- {item}" for item in prior[-3:])
        return (
            CFQ_REASONING_WITH_PRIOR.format(prior=prior_lines, current=current or "(none)"),
            True,
        )

    def generate(
        self,
        *,
        reasoning: str,
        command: str,
        prior_reasoning: list[str] | None = None,
        existing_cfq: str | None = None,
    ) -> str | None:
        if existing_cfq and not self.config.refine_existing:
            return existing_cfq
        if not reasoning.strip() and not prior_reasoning:
            return existing_cfq

        reasoning_block, used_prior = self.build_reasoning_block(
            reasoning,
            prior_reasoning,
            min_chars=self.config.min_reasoning_chars,
        )

        messages = [
            {"role": "system", "content": CFQ_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CFQ_USER_TEMPLATE.format(
                    reasoning_block=reasoning_block[:8000],
                    command=command,
                ),
            },
        ]
        if existing_cfq and self.config.refine_existing:
            messages.append(
                {
                    "role": "user",
                    "content": f"Refine this draft CFQ into a better pruner query:\n{existing_cfq}",
                }
            )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.config.chat_template_kwargs

        last_error: Exception | None = None
        for _ in range(max(self.config.retries, 1)):
            try:
                response = self.session.post(self.config.url, json=payload, timeout=self.config.timeout)
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                text = (message.get("content") or message.get("reasoning") or "").strip()
                return self._normalize_output(text)
            except Exception as exc:
                last_error = exc
                logger.debug("CFQ generator request failed: %s", exc)
        if last_error:
            logger.warning("CFQ generator failed after retries: %s", last_error)
        return existing_cfq

    @staticmethod
    def _normalize_output(text: str) -> str | None:
        text = re.sub(r"</?context_focus_question>", "", text, flags=re.IGNORECASE).strip()
        if not text or text.upper() == "SKIP":
            return None
        text = text.split("\n\n", 1)[0].strip()
        # Reject questions that leak file/line hints into the pruner query.
        if re.search(r"\b(lines?\s+\d|line\s+numbers?|at lines?\s+\d|\d+\s*-\s*\d+\s+in\b)", text, re.I):
            return None
        return text or None
