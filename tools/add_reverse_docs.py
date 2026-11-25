"""Add the Hindi -> English documentation to the README (safe to run more than once).

    python tools/add_reverse_docs.py
"""
import re
import sys
from pathlib import Path

START, END = "<!-- REVERSE:START -->", "<!-- REVERSE:END -->"

SECTION = f"""{START}
## Hindi to English (reverse direction)

The same pipeline also dubs **Indian-language speech into English**. English output can be
checked by any listener, which makes this direction easy to verify.

- **Models:** speech recognition is the same Whisper model. Translation uses the matching
  `ai4bharat/indictrans2-indic-en-dist-200M` (gated, same Hugging Face login, but you must accept
  its access terms on its own page). English speech comes from Indic Parler-TTS, which supports
  English (speaker *Mary*).
- **Download once:**
  ```powershell
  hf download ai4bharat/indictrans2-indic-en-dist-200M --local-dir .\\models\\indictrans2-indic-en
  ```
- **Command line:** `python -m dubstudio .\\samples\\hindi_talk.wav --src hi --lang en`
- **Web UI:** choose *Hindi to English* in the Direction menu.
- **Agent:** English is usually shorter than Hindi, so far fewer lines overrun. When one does, the
  same n-best and LLM rewrite steps apply; the meaning check compares the English rewrite
  directly with the Hindi source using a multilingual embedding.
- **Supported directions:** English to any supported Indian language, and any supported Indian
  language to English. Indian language to Indian language is refused.
- **Status:** Hindi to English is new and lightly tested. Other source languages are experimental.
{END}
"""


def apply(text):
    """Return ``text`` with the reverse-direction section inserted or refreshed."""
    if START in text:
        return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: SECTION.strip(),
                      text, flags=re.S)
    marker = "## Results"
    text = text.replace(marker, SECTION + "\n" + marker, 1) if marker in text else text + "\n" + SECTION
    text = text.replace("- English source speech only. Other languages are refused with a clear message.",
                        "- Sources: English, and Hindi to English (new, lightly tested). Other source "
                        "languages are refused with a clear message.")
    text = text.replace("| Translation | `ai4bharat/indictrans2-en-indic-dist-200M` (gated: Hugging Face login needed) |",
                        "| Translation | `ai4bharat/indictrans2-en-indic-dist-200M`, and `ai4bharat/indictrans2-indic-en-dist-200M` "
                        "for Indian language to English (both gated: Hugging Face login needed) |")
    text = text.replace(
        "- **Five target languages:** Hindi (checked by a fluent listener), plus Marathi, Bengali, Tamil and Telugu (experimental, unchecked).",
        "- **Both directions:** English to Hindi (checked by a fluent listener) and to Marathi, Bengali, "
        "Tamil and Telugu (experimental), plus Hindi to English (new, lightly tested).")
    return text


def count_tests(root="."):
    """Number of test methods under ``tests/`` (matches the unittest run count)."""
    return sum(len(re.findall(r"^\s+def test_", f.read_text(encoding="utf-8"), re.M))
               for f in (Path(root) / "tests").glob("test_*.py"))


def update_counts(text, n):
    """Replace every 'N unit tests' with the real count."""
    return re.sub(r"\b\d+ unit tests\b", f"{n} unit tests", text)


def main(path="README.md"):
    p = Path(path)
    text = update_counts(apply(p.read_text(encoding="utf-8")), count_tests(p.parent))
    p.write_text(text, encoding="utf-8")
    print(f"Updated {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
