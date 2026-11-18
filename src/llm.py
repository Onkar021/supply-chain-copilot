"""Thin, swappable LLM wrapper. Priority: Anthropic API -> local Ollama -> None (deterministic fallback).
Every caller must work when complete() returns None - that is what makes the system demo-proof."""
import os
import requests
from . import config as C


class LLM:
    def __init__(self):
        self.backend, self.name = None, "none (deterministic fallback)"
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                self.backend, self.name = "anthropic", C.ANTHROPIC_MODEL
            except ImportError:
                print("[llm] anthropic package missing: pip install anthropic")
        if self.backend is None and os.getenv("USE_OLLAMA", "0") == "1":
            try:
                requests.get(C.OLLAMA_URL, timeout=2)
                self.backend, self.name = "ollama", C.OLLAMA_MODEL
            except Exception:
                print("[llm] Ollama not reachable")

    @property
    def available(self): return self.backend is not None

    def complete(self, system: str, user: str, max_tokens=700, temperature=0.3):
        try:
            if self.backend == "anthropic":
                r = self.client.messages.create(model=C.ANTHROPIC_MODEL, max_tokens=max_tokens, temperature=temperature,
                                                system=system, messages=[{"role": "user", "content": user}])
                return r.content[0].text
            if self.backend == "ollama":
                r = requests.post(f"{C.OLLAMA_URL}/api/chat", timeout=120, json={
                    "model": C.OLLAMA_MODEL, "stream": False, "options": {"temperature": temperature},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
                return r.json()["message"]["content"]
        except Exception as e:
            print(f"[llm] call failed ({type(e).__name__}: {e}) -> falling back")
        return None
