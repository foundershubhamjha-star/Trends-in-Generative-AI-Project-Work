"""Put the two video links into the README.

    python tools/set_links.py --demo "https://drive.google.com/..." --walkthrough "https://drive.google.com/..."

Adds a "Videos" block before the Features section (or updates it if already present).
"""
import argparse
import re
import sys
from pathlib import Path

START, END = "<!-- VIDEOS:START -->", "<!-- VIDEOS:END -->"


def valid(url):
    """Accept only Google Drive links."""
    return bool(re.match(r"https://(drive|docs)\.google\.com/\S+$", url))


def build_block(demo, walkthrough):
    """The markdown block with both links."""
    return (f"{START}\n**Demo video:** [{demo}]({demo})  \n"
            f"**Code walkthrough video:** [{walkthrough}]({walkthrough})\n{END}")


def apply(text, demo, walkthrough):
    """Return ``text`` with the video block inserted or replaced."""
    block = build_block(demo, walkthrough)
    if START in text and END in text:
        return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: block, text, flags=re.S)
    marker = "## Features"
    if marker in text:
        return text.replace(marker, f"## Videos\n\n{block}\n\n{marker}", 1)
    return text.rstrip() + f"\n\n## Videos\n\n{block}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--walkthrough", required=True)
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()
    for label, url in (("demo", args.demo), ("walkthrough", args.walkthrough)):
        if not valid(url):
            print(f"The {label} link must be a Google Drive link (https://drive.google.com/...).")
            return 1
    p = Path(args.readme)
    p.write_text(apply(p.read_text(encoding="utf-8"), args.demo, args.walkthrough), encoding="utf-8")
    print(f"Updated {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
