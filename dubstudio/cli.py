"""Command-line interface: ``python -m dubstudio INPUT [INPUT ...] --lang hi``."""
import argparse
import csv
import glob
import logging
from pathlib import Path

from . import config
from .errors import DubStudioError
from .pipeline import run

CSV_FIELDS = ["input", "language", "agent", "lines", "fixable_before", "fixed", "fixable_after",
              "artifacts", "mean_tempo_before", "mean_tempo_after", "max_tempo_before",
              "max_tempo_after", "seconds"]


def _expand(patterns):
    """Expand wildcards ourselves (PowerShell does not expand them for native programs)."""
    files = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        files.extend(hits if hits else [p])
    return files


def _append_csv(path, summary):
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**summary, "language": summary.get("language")})


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="dubstudio", description="Dub speech (audio or video) between English and an Indian language.")
    ap.add_argument("inputs", nargs="+", help="input files; wildcards allowed")
    ap.add_argument("--lang", default="hi", choices=sorted(config.LANGUAGES),
                    help="target language (English and Hindi have been quality-checked)")
    ap.add_argument("--src", default="en", choices=sorted(config.LANGUAGES),
                    help="spoken language of the input (default en). For Hindi to English use "
                         "--src hi --lang en")
    ap.add_argument("--no-agent", action="store_true", help="skip the shortening agent")
    ap.add_argument("--thr", type=float, default=config.DEFAULT_THR,
                    help="allowed speech-to-slot ratio before the agent acts")
    ap.add_argument("--format", choices=["wav", "mp3"], default="wav", help="audio output format")
    ap.add_argument("--skip-language-check", action="store_true",
                    help="do not refuse non-English source audio")
    ap.add_argument("--out", default=None, help="output directory (default: outputs/)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    failures = 0
    out_dir = Path(args.out) if args.out else config.out_dir()
    for f in _expand(args.inputs):
        print(f"\n== {f}")
        try:
            res = run(f, lang=args.lang, use_agent=not args.no_agent, thr=args.thr,
                      audio_format=args.format, check_language=not args.skip_language_check,
                      source=args.src,
                      out_root=out_dir,
                      progress=lambda stage, frac, msg: print(f"  [{frac:4.0%}] {stage} {msg}"))
        except DubStudioError as exc:
            failures += 1
            print(f"  FAILED: {exc}")
            continue
        for w in res["warnings"]:
            print(f"  note: {w}")
        for tag, path in res["outputs"].items():
            print(f"  {tag}: {path}")
        s = res["summary"]
        print(f"  lines={s['lines']} fixable_before={s['fixable_before']} fixed={s['fixed']} "
              f"artifacts={s['artifacts']} mean_speedup {s['mean_tempo_before']} -> "
              f"{s['mean_tempo_after']} ({s['seconds']} s)")
        out_dir.mkdir(parents=True, exist_ok=True)
        _append_csv(out_dir / "results.csv", s)
    return 1 if failures else 0
