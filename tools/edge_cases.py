"""Robustness suite: feed the pipeline awkward inputs and check it fails *gracefully*.

    python tools/edge_cases.py --base samples/clip2.wav          # build inputs, run, report
    python tools/edge_cases.py --base samples/clip2.wav --inject README.md

Every case has an expected outcome: either success, or a specific user-facing error
(``InputError``, ``NoSpeechError``, ``UnsupportedLanguageError``). A case PASSES when the
outcome matches. Any other exception (a crash) is a FAIL. ``--base`` must be a short clip of
clean English speech; the cases are derived from it. ``--hindi`` is a Hindi dub made earlier
(used as a non-English source), e.g. ``outputs/clip2_hi_agent.wav``.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools._inject import inject  # noqa: E402

OK = "ok"


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def build_inputs(base, hindi, out):
    """Create the test files. Returns a list of case dicts."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sr = 16000
    speech, ssr = sf.read(str(base), dtype="float32")
    if speech.ndim == 2:
        speech = speech.mean(axis=1)

    def wav(name, y, rate):
        p = out / name
        sf.write(p, y, rate)
        return p

    cases = []

    def add(name, what, path, expect, lang="hi", source="en"):
        cases.append({"name": name, "what": what, "path": Path(path), "expect": expect,
                      "lang": lang, "source": source})

    add("silence", "10 s of digital silence", wav("silence.wav", np.zeros(sr * 10, "float32"), sr),
        "NoSpeechError")
    t = np.arange(sr * 12) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330)) * 0.1
    add("music_only", "synthetic chord, no speech", wav("music_only.wav", chord.astype("float32"), sr),
        "NoSpeechError")
    add("white_noise", "10 s of noise", wav("white_noise.wav", (0.2 * rng.standard_normal(sr * 10)).astype("float32"), sr),
        "NoSpeechError")
    short = speech[: int(3 * ssr)]
    add("short_3s", "3-second excerpt of speech", wav("short_3s.wav", short, ssr), OK)
    noise = rng.standard_normal(len(speech)).astype("float32")
    snr_gain = np.sqrt((speech ** 2).mean() / (noise ** 2).mean() / 10 ** (10 / 10))  # 10 dB SNR
    add("noisy_10dB", "speech with white noise at 10 dB SNR", wav("noisy_10dB.wav", speech + snr_gain * noise, ssr), OK)
    ffmpeg("-i", base, "-filter:a", "volume=0.05", out / "very_quiet.wav")
    add("very_quiet", "speech at -26 dB", out / "very_quiet.wav", OK)
    ffmpeg("-i", base, "-filter:a", "atempo=1.5", out / "fast_speech.wav")
    add("fast_speech", "speech sped up 1.5x", out / "fast_speech.wav", OK)
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=15", "-i", base, "-shortest",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out / "video_with_speech.mp4")
    add("video_with_speech", "MP4 video + speech", out / "video_with_speech.mp4", OK)
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=15", "-t", "5", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", out / "video_no_audio.mp4")
    add("video_no_audio", "video with no audio track", out / "video_no_audio.mp4", "InputError")
    if hindi and Path(hindi).exists():
        add("hindi_source", "Hindi speech given as the source", hindi, "UnsupportedLanguageError")
        add("reverse_hi_to_en", "Hindi speech dubbed into English (needs the reverse model)",
            hindi, OK, lang="en", source="hi")
    add("english_as_hindi", "English speech with source set to Hindi", base,
        "UnsupportedLanguageError", lang="en", source="hi")
    add("same_language", "English to English", base, "InputError", lang="en", source="en")
    add("indic_to_indic", "Hindi to Marathi (unsupported direction)", base, "InputError",
        lang="mr", source="hi")
    (out / "corrupt.mp3").write_bytes(rng.bytes(5000))
    add("corrupt_file", "random bytes named .mp3", out / "corrupt.mp3", "InputError")
    (out / "empty.wav").write_bytes(b"")
    add("empty_file", "zero-byte file", out / "empty.wav", "InputError")
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "1860", out / "too_long.wav")
    add("too_long_31min", "31 minutes (limit is 30)", out / "too_long.wav", "InputError")
    add("missing_file", "path that does not exist", out / "does_not_exist.wav", "InputError")
    add("bad_language", "unsupported target language code 'xx'", base, "InputError", lang="xx")
    return cases


def run_case(case, runner):
    """Run one case; return ``(outcome, detail)``. Never raises."""
    t0 = time.time()
    try:
        res = runner(str(case["path"]), lang=case["lang"], source=case.get("source", "en"),
                     use_agent=True)
        s = res["summary"]
        return OK, f"{s['lines']} lines, {time.time() - t0:.0f} s"
    except Exception as exc:  # noqa: BLE001 - the point is to catch everything
        return type(exc).__name__, str(exc)[:110]


def to_markdown(rows):
    """Render result rows as a markdown table. ``rows``: dicts with name/what/expect/got/detail."""
    out = ["| Case | What it tests | Expected | Got | Result |", "|---|---|---|---|---|"]
    for r in rows:
        status = "PASS" if r["expect"] == r["got"] else "FAIL"
        out.append(f"| `{r['name']}` | {r['what']} | {r['expect']} | {r['got']} | {status} |")
    passed = sum(1 for r in rows if r["expect"] == r["got"])
    out.append("")
    out.append(f"{passed} / {len(rows)} cases behaved as expected.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="clean English speech clip (a few seconds to minutes)")
    ap.add_argument("--hindi", default=None, help="a Hindi audio file to use as a non-English source")
    ap.add_argument("--dir", default="samples/edge")
    ap.add_argument("--out", default="outputs/edge_cases.md")
    ap.add_argument("--inject", default=None, help="README to update between the EDGE markers")
    args = ap.parse_args()

    from dubstudio.pipeline import run
    cases = build_inputs(Path(args.base), args.hindi, Path(args.dir))
    rows = []
    for c in cases:
        print(f"-> {c['name']}: {c['what']}", flush=True)
        got, detail = run_case(c, run)
        mark = "PASS" if got == c["expect"] else "FAIL"
        print(f"   {mark}  expected={c['expect']} got={got}  {detail}", flush=True)
        rows.append({**{k: c[k] for k in ("name", "what", "expect")}, "got": got, "detail": detail})
    md = to_markdown(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    if args.inject:
        inject(args.inject, "EDGE", md)
        print(f"\nUpdated {args.inject}")
    return 0 if all(r["expect"] == r["got"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
