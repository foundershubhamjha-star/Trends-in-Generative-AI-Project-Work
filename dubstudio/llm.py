"""Minimal client for a local Ollama server."""
import requests

from . import config


class LLM:
    """Chat client. Use :meth:`available` first so the pipeline can degrade gracefully."""

    def __init__(self, model=None, url=None):
        self.model = model or config.LLM_MODEL
        self.url = (url or config.OLLAMA_URL).rstrip("/")

    def available(self) -> bool:
        """True if the server is reachable and the model has been pulled."""
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(n == self.model or n.startswith(self.model) for n in names)
        except Exception:
            return False

    def chat(self, prompt, temperature=0.2, max_tokens=400) -> str:
        """Send one user prompt and return the reply text."""
        r = requests.post(f"{self.url}/api/chat", json={
            "model": self.model, "stream": False, "keep_alive": "10m",
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": "You are a precise dubbing editor. "
                                              "Output only what is asked, no explanations."},
                {"role": "user", "content": prompt}]}, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip().strip('"').strip()
