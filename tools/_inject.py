"""Replace the text between ``<!-- NAME:START -->`` and ``<!-- NAME:END -->`` in a file."""
import re
from pathlib import Path


def inject(path, name, text):
    """Insert ``text`` between the NAME markers of ``path``. Raises ValueError if absent."""
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.S)
    if not pat.search(src):
        raise ValueError(f"markers for {name} not found in {path}")
    p.write_text(pat.sub(lambda m: f"{m.group(1)}\n{text}\n{m.group(3)}", src), encoding="utf-8")
