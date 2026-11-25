"""Final checklist for the submission: README, licence, video links, file sizes.

    python tools/check_submission.py

Exit code 1 if anything FAILs. WARN items deserve a look but do not block.
"""
import re
import sys
from pathlib import Path

SKIP_DIRS = {"venv", "models", "work", "samples", "outputs", ".git", "__pycache__"}
SIZE_LIMIT = 25 * 1024 * 1024  # GitHub's web uploader limit per file


def check_readme(text):
    """Checks on README content. Returns a list of (status, message)."""
    out = []
    out.append(("FAIL", "results table is still a placeholder") if "_Run the evaluation" in text
               else ("OK", "results table filled"))
    out.append(("FAIL", "edge-case table is still a placeholder") if "_Run the edge-case suite" in text
               else ("OK", "edge-case table filled"))
    if re.search(r"\|\s*FAIL\s*\|", text):
        out.append(("WARN", "edge-case table contains FAIL rows: fix them or explain them in the README"))
    links = re.findall(r"https://drive\.google\.com/\S+", text)
    out.append(("OK", f"{len(set(links))} Drive link(s) in README") if len(set(links)) >= 2
               else ("FAIL", "README needs two Google Drive links (python tools/set_links.py ...)"))
    out.append(("OK", "no leftover TODO/TBD") if not re.search(r"\b(TODO|TBD|XXX)\b", text)
               else ("WARN", "README contains TODO/TBD text"))
    return out


def check_repo(root):
    """Checks on repository files."""
    root = Path(root)
    out = []
    lic = root / "LICENSE"
    if not lic.exists():
        out.append(("FAIL", "LICENSE missing"))
    elif "YOUR NAME" in lic.read_text(encoding="utf-8"):
        out.append(("FAIL", "LICENSE still says YOUR NAME"))
    else:
        out.append(("OK", "LICENSE present"))
    for name in ("README.md", "requirements.txt", ".gitignore", "app.py"):
        out.append(("OK", f"{name} present") if (root / name).exists() else ("FAIL", f"{name} missing"))
    for d in ("dubstudio", "tests", "docs"):
        out.append(("OK", f"{d}/ present") if (root / d).is_dir() else ("FAIL", f"{d}/ missing"))
    big = []
    for p in root.rglob("*"):
        if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS) and p.stat().st_size > SIZE_LIMIT:
            big.append(p.name)
    out.append(("WARN", "files over 25 MB (web upload will refuse): " + ", ".join(big)) if big
               else ("OK", "no oversized files"))
    present = [d for d in ("venv", "models", "work", "samples", "outputs") if (root / d).exists()]
    if present:
        out.append(("WARN", "do NOT upload these folders to GitHub: " + ", ".join(present)))
    return out


def main(root="."):
    root = Path(root)
    results = []
    readme = root / "README.md"
    if readme.exists():
        results += check_readme(readme.read_text(encoding="utf-8"))
    results += check_repo(root)
    for status, msg in results:
        print(f"[{status:4}] {msg}")
    bad = sum(1 for s, _ in results if s == "FAIL")
    print(f"\n{'READY TO SUBMIT' if not bad else str(bad) + ' item(s) to fix'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
