# llm_integration.py

from dataclasses import dataclass
from typing import Literal, List

import os

from config import LLM_PROVIDER, OPENAI_MODEL, OLLAMA_MODEL

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import requests


@dataclass
class LLMConfig:
    provider: Literal["openai", "ollama", "template"] = LLM_PROVIDER
    openai_model: str = OPENAI_MODEL
    ollama_model: str = OLLAMA_MODEL


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._openai = None
        if config.provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed.")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY env var required for OpenAI provider.")
            self._openai = OpenAI(api_key=api_key)

    def _build_system_prompt(self) -> str:
        return (
            "You are an HR policy assistant focused on employee wellbeing, burnout "
            "prevention, and performance support. Use only the provided policy context. "
            "If something is not covered by policy, say so explicitly. Be clear, concise, "
            "and practical."
        )

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        if self.config.provider == "openai":
            return self._call_openai(question, context_chunks)
        elif self.config.provider == "ollama":
            return self._call_ollama(question, context_chunks)
        else:
            return self._template_answer(question, context_chunks)

    def _build_context_block(self, context_chunks: List[str]) -> str:
        joined = "\n\n---\n\n".join(context_chunks)
        return f"HR POLICY CONTEXT:\n{joined}\n\nUSER QUESTION:"

    # --- Providers ---

    def _call_openai(self, question: str, context_chunks: List[str]) -> str:
        content = (
            self._build_context_block(context_chunks)
            + f"\n\n{question}\n\nAnswer using only the context above."
        )
        resp = self._openai.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def _call_ollama(self, question: str, context_chunks: List[str]) -> str:
        content = (
            self._build_context_block(context_chunks)
            + f"\n\n{question}\n\nAnswer using only the context above."
        )
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.config.ollama_model,
                "messages": [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": content},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()

    def _template_answer(self, question: str, context_chunks: List[str]) -> str:
        # Dumb fallback, but deterministic.
        bullet_points = []
        for i, chunk in enumerate(context_chunks[:3], 1):
            bullet_points.append(f"- From policy snippet {i}: {chunk[:200]}...")
        return (
            "Here’s a synthesized answer based on your HR policies.\n\n"
            + "\n".join(bullet_points)
            + "\n\nQuestion:\n"
            + question
            + "\n\n(Template mode: plug in a real LLM to get proper natural language answers.)"
        )
