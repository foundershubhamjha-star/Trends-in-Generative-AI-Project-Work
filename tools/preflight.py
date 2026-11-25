"""Check that this machine is ready to dub: GPU, ffmpeg, models, Ollama, libraries.

    python tools/preflight.py

Run it before recording the videos and again before the review.
"""
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubstudio import config  # noqa: E402


def check_models(models_dir):
    """Return (status, message) for every required model folder."""
    out = []
    for name in (config.WHISPER_DIR, config.TRANSLATE_DIR, config.TTS_DIR, config.EMBED_DIR):
        p = Path(models_dir) / name
        ok = p.is_dir() and (p / "config.json").is_file() or name == config.WHISPER_DIR and p.is_dir() and any(p.iterdir())
        out.append(("OK" if ok else "FAIL", f"model folder {p}" + ("" if ok else " is missing or empty")))
    return out


def check_modules(names):
    """Return (status, message) for every importable library."""
    out = []
    for n in names:
        try:
            importlib.import_module(n)
            out.append(("OK", f"python package {n}"))
        except Exception as exc:  # noqa: BLE001
            out.append(("FAIL", f"python package {n}: {type(exc).__name__}"))
    return out


def check_tools(names):
    """Return (status, message) for every command-line tool on PATH."""
    return [("OK" if shutil.which(n) else "FAIL",
             f"command {n}" + ("" if shutil.which(n) else " not found on PATH")) for n in names]


def check_gpu():
    """CUDA availability and free memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return [("FAIL", "CUDA is not available to PyTorch")]
        free, total = torch.cuda.mem_get_info()
        gb = free / 1e9
        status = "OK" if gb >= 8 else "WARN"
        return [(status, f"GPU {torch.cuda.get_device_name(0)}: {gb:.1f} GB free of {total / 1e9:.1f} GB"
                         + ("" if status == "OK" else " (close Chrome/Firefox and other GPU apps)"))]
    except Exception as exc:  # noqa: BLE001
        return [("FAIL", f"GPU check failed: {exc}")]


def check_ollama():
    """Ollama server and model (needed for the agent's LLM step; optional)."""
    from dubstudio.llm import LLM
    ok = LLM().available()
    return [("OK" if ok else "WARN",
             f"Ollama model {config.LLM_MODEL}" + ("" if ok else
             " not reachable: start Ollama; the LLM rewrite step will be skipped without it"))]


def main():
    results = []
    results += check_tools(["ffmpeg", "ffprobe"])
    results += check_modules(["torch", "faster_whisper", "transformers", "parler_tts",
                              "sentence_transformers", "gradio", "demucs"])
    results += check_gpu()
    results += check_models(config.models_dir())
    rev = Path(config.models_dir()) / config.TRANSLATE_REV_DIR
    rev_ok = (rev / "config.json").is_file()
    results.append(("OK" if rev_ok else "WARN",
                    f"reverse model folder {rev}"
                    + ("" if rev_ok else " missing or incomplete: needed only for Hindi -> English")))
    results += check_ollama()
    for status, msg in results:
        print(f"[{status:4}] {msg}")
    bad = sum(1 for s, _ in results if s == "FAIL")
    print(f"\n{'READY' if not bad else str(bad) + ' problem(s) to fix'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
