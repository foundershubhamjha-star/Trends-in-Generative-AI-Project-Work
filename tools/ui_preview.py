"""Launch the web UI with a fake pipeline (no GPU, no models) to preview the layout.

    python tools/ui_preview.py
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dubstudio import ui  # noqa: E402
from dubstudio.segments import summarize  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dubstudio_preview_"))


def _tone(name, seconds, freq):
    t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False)
    path = TMP / name
    sf.write(path, (0.2 * np.sin(2 * np.pi * freq * t)).astype("float32"), 16000)
    return str(path)


def fake_run(path, lang="hi", use_agent=True, thr=1.15, progress=None, **_):
    for i, stage in enumerate(["audio", "separate", "transcribe", "translate", "synthesize", "agent"]):
        if progress:
            progress(stage, (i + 1) / 7, "")
        time.sleep(0.2)
    mk = lambda s, e, en, tg, raw, av: dict(start=s, end=e, text=en, text_tgt=tg, raw_dur=raw,
                                            avail=av, wav="x")
    before = [mk(0, 4, "Hello everyone.", "सभी को नमस्कार।", 2.0, 4.5),
              mk(4.5, 8, "We hold these truths to be self-evident.", "हम इन सच्चाइयों को स्वयंसिद्ध मानते हैं।", 6.2, 4.4),
              mk(8, 8.4, "Yes.", "हाँ।", 1.0, 0.45)]
    after = [dict(b) for b in before]
    after[1].update(text_tgt="हम इन सच्चाइयों को स्पष्ट मानते हैं।", raw_dur=4.4)
    summary = summarize(before, after, thr)
    summary.update(language=lang, seconds=63.0, input=Path(path).name, agent=use_agent)
    trace = [{"line": 1, "english": before[1]["text"], "before_dur": 6.2, "avail": 4.4,
              "after_dur": 4.4, "fixed": True,
              "steps": [{"strategy": "nbest", "chars": 36, "raw_dur": 4.4, "text": "...", "ok_time": True}]}]
    outs = {"baseline": _tone("base.wav", 8, 330), "agent": _tone("agent.wav", 8, 440)}
    ctx = {"input": path, "background": "", "has_video": False, "stem": "demo", "lang": lang,
           "thr": thr, "audio_format": "wav", "out_dir": str(TMP)}
    return {"outputs": outs, "summary": summary, "lines": after, "baseline_lines": before,
            "trace": trace, "warnings": [] if lang == "hi" else ["Experimental language."],
            "work_dir": str(TMP), "context": ctx}


def fake_redub(result, texts, progress=None):
    time.sleep(0.5)
    changed = [i for i, t in enumerate(texts) if t != result["lines"][i]["text_tgt"]]
    lines = [dict(l, text_tgt=t) for l, t in zip(result["lines"], texts)]
    return {"outputs": {"edited": _tone("edited.wav", 8, 550)}, "lines": lines, "changed": changed}


if __name__ == "__main__":
    ui.launch(fake_run, fake_redub, inbrowser=False)
