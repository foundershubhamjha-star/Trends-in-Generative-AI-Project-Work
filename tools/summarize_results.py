"""Turn ``outputs/results.csv`` (written by the CLI) into a markdown table.

    python tools/summarize_results.py                       # print the table
    python tools/summarize_results.py --inject README.md    # write it into the README
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools._inject import inject  # noqa: E402


def load_latest(path):
    """Read the CSV keeping only the most recent agent run per (input, language)."""
    latest = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("agent", "True") in ("True", "true", "1"):
                latest[(row["input"], row["language"])] = row
    return list(latest.values())


def to_markdown(rows):
    """Render rows plus a totals line."""
    head = ["| File | Lines | Too long | Fixed | Still long | Timing artifacts | Mean speed-up | Time |",
            "|---|---|---|---|---|---|---|---|"]
    body = []
    for r in rows:
        body.append(f"| {r['input']} | {r['lines']} | {r['fixable_before']} | {r['fixed']} | "
                    f"{r['fixable_after']} | {r['artifacts']} | "
                    f"{float(r['mean_tempo_before']):.2f}x → {float(r['mean_tempo_after']):.2f}x | "
                    f"{float(r['seconds']):.0f} s |")
    tf = sum(int(r["fixable_before"]) for r in rows)
    fx = sum(int(r["fixed"]) for r in rows)
    foot = f"\n**Total:** the agent fixed {fx} of {tf} too-long lines across {len(rows)} files."
    return "\n".join(head + body) + foot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="outputs/results.csv")
    ap.add_argument("--inject", default=None, help="README to update between the RESULTS markers")
    args = ap.parse_args()
    rows = load_latest(args.csv)
    if not rows:
        print("No agent runs found in", args.csv)
        return 1
    md = to_markdown(rows)
    print(md)
    if args.inject:
        inject(args.inject, "RESULTS", md)
        print(f"\nUpdated {args.inject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
