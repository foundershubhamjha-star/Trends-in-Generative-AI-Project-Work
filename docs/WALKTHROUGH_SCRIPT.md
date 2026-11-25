# Code walkthrough video script (about 6 minutes)

Screen: VS Code with the repo open, then PowerShell.

**1. Setup (1 min).** Show `requirements.txt` and say why `transformers` is pinned (parler-tts). Show the `hf download` lines in the README and the `models/` folder. Say what Ollama is used for.

**2. Folder tour (1 min).** Walk down the README layout: `dubstudio/` (one module per stage), `tests/`, `tools/`, `docs/`, `legacy/`. Say: "`segments.py` holds all the pure logic, with no heavy imports, so it is unit-tested without a GPU."

**3. The pipeline (1.5 min).** Open `pipeline.py`, `run()`. Walk the numbered comments: validate, audio, separate, transcribe, merge, translate, synthesize, agent, render. Point out the on-disk caching and the `try/finally` that frees GPU memory. Open `errors.py` and show how the UI and CLI turn those into messages.

**4. The agent (1.5 min).** Open `agent.py`. Show the module docstring, then `_nbest` (cheap, no LLM) and `_llm_rewrite` (length floor, back-translation similarity, judge, feedback on retry, rising temperature). Say: "A candidate is only accepted if it is strictly shorter in spoken duration, so the agent cannot make a line worse."

**5. Tests (30 s).** In PowerShell run `python -m unittest discover -s tests -t .` and show `OK`. Open one agent test that uses fake models.

**6. UI and tools (30 s).** Open `ui.py` (dependency injection of `run`/`redub`), then `tools/edge_cases.py`. Say what each does in one sentence.
