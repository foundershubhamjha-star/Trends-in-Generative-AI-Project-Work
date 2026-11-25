"""End-to-end orchestration: input file -> dubbed audio/video.

Stages (each cached on disk under ``work/<name>/`` so reruns skip finished work):
validate -> extract audio -> separate vocals -> transcribe -> merge segments -> translate
-> synthesize -> (agent) -> render.
"""
import copy
import json
import logging
import re
import time
from pathlib import Path

from . import asr, audio, config, gpu
from .agent import AgentConfig, ShorteningAgent, make_similarity
from .errors import DubStudioError, InputError, NoSpeechError, UnsupportedLanguageError
from .llm import LLM
from .segments import (available_time, cache_key, classify, merge_segments, summarize,
                       tempo_for)
from .translate import Translator
from .tts import Synthesizer

log = logging.getLogger(__name__)


def _slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "input"


def _dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _render(src, has_video, lines, bg, wd, out_dir, stem, lang, tag, audio_format):
    """Time-fit every clip, mix over the background and write the final file."""
    fit_dir = Path(wd) / "fit" / tag
    clips = []
    for i, ln in enumerate(lines):
        dst = fit_dir / f"{i:04d}.wav"
        audio.speed_up(ln["wav"], dst, tempo_for(ln["raw_dur"], ln["avail"]))
        clips.append((ln["start"], dst))
    mixed = Path(wd) / f"mix_{tag}.wav"
    audio.mix(bg, clips, mixed)
    if has_video:
        out = Path(out_dir) / f"{stem}_{lang}_{tag}.mp4"
        audio.mux(src, mixed, out)
    else:
        out = Path(out_dir) / f"{stem}_{lang}_{tag}.{audio_format}"
        audio.export_audio(mixed, out, audio_format)
    return out


def run(input_path, lang="hi", use_agent=True, thr=config.DEFAULT_THR, progress=None,
        audio_format="wav", check_language=True, max_minutes=config.MAX_INPUT_MINUTES,
        work_root=None, out_root=None, source="en"):
    """Dub one audio or video file.

    Args:
        input_path: audio (mp3/wav/m4a/flac) or video file with English speech.
        lang: target language code, see :data:`dubstudio.config.LANGUAGES`.
        source: spoken language of the input. ``en`` (default) with any Indian target, or an
            Indian language with ``lang="en"`` (see :func:`dubstudio.config.get_direction`).
        use_agent: run the shortening agent (also renders the un-fixed baseline for comparison).
        thr: allowed ratio of speech duration to available time before the agent acts.
        progress: optional callback ``progress(stage, fraction, message)``.
        audio_format: ``wav`` or ``mp3`` for audio-only inputs.
        check_language: refuse audio whose detected language differs from ``source``.
        max_minutes: reject inputs longer than this.
        work_root, out_root: override the work and output directories.

    Returns:
        dict with ``outputs`` (paths), ``summary`` (metrics), ``lines``, ``baseline_lines``,
        ``trace`` (agent steps), ``warnings`` and ``work_dir``.

    Raises:
        DubStudioError subclasses with user-presentable messages.
    """
    def report(stage, frac, msg=""):
        log.info("%s %s", stage, msg)
        if progress:
            progress(stage, frac, msg)

    t0 = time.time()
    warnings = []
    direction = config.get_direction(source, lang)
    language = direction.target
    if not language.verified:
        warnings.append(f"{language.name} output has not been quality-checked (experimental).")
    if not direction.source.verified:
        warnings.append(f"{direction.source.name} input has not been quality-checked (experimental).")

    work_root = Path(work_root or config.work_dir())
    out_root = Path(out_root or config.out_dir())
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 1. validate input ----
    src = Path(input_path)
    if not src.is_file():
        raise InputError(f"File not found: {src}")
    info = audio.probe(src)
    if not info["has_audio"]:
        raise InputError("This file has no audio track.")
    if info["duration"] < 1.0:
        raise InputError("The audio is shorter than one second.")
    if info["duration"] > max_minutes * 60:
        raise InputError(f"The audio is longer than {max_minutes} minutes "
                         f"({info['duration'] / 60:.1f} min). Please split it first.")
    stem = _slug(src.stem)
    wd = work_root / stem
    ld = wd / (lang if source == "en" else f"{source}-{lang}")
    ld.mkdir(parents=True, exist_ok=True)

    model_path = config.models_dir() / direction.model_dir
    if not (model_path / "config.json").is_file():
        hint = ("" if source == "en" else
                " Download it with: hf download ai4bharat/indictrans2-indic-en-dist-200M "
                f"--local-dir .\\models\\{direction.model_dir}")
        raise DubStudioError(f"Translation model is missing or incomplete: {model_path} "
                             f"(no config.json).{hint}")

    # ---- 2. audio + vocal separation ----
    report("audio", 0.05, "extracting audio")
    wav = wd / "audio.wav"
    if not wav.exists():
        audio.extract_audio(src, wav)
    report("separate", 0.10, "separating speech from background")
    vocals, background = audio.separate_vocals(wav, wd / "sep")

    # ---- 3. transcription ----
    report("transcribe", 0.25, "transcribing speech")
    asr_file = wd / "asr.json"
    if asr_file.exists():
        data = _load(asr_file)
    else:
        segs, detected, prob = asr.transcribe(vocals, config.models_dir() / config.WHISPER_DIR, wd)
        data = {"language": detected, "probability": prob, "segments": segs}
        _dump(asr_file, data)
    if not data["segments"]:
        raise NoSpeechError("No speech was detected (silence, music only, or too noisy).")
    if check_language and data["language"] != source and data["probability"] >= 0.6:
        raise UnsupportedLanguageError(
            f"The audio looks like '{data['language']}' (confidence {data['probability']:.0%}), "
            f"but the source language is set to {direction.source.name}. "
            "Change the source language, or this audio is not supported.")
    segs = merge_segments(data["segments"])

    translator = synth = None
    try:
        # ---- 4. translation ----
        report("translate", 0.40, f"translating to {language.name}")
        translator = Translator(config.models_dir(), language.indictrans, direction.model_dir,
                                direction.source.indictrans)
        key_parts = ([language.indictrans] if source == "en"      # keeps old English->X caches valid
                     else [direction.source.indictrans, language.indictrans])
        tfile = ld / f"translation_{cache_key(*key_parts, *[s['text'] for s in segs])}.json"
        if tfile.exists():
            texts = _load(tfile)
        else:
            texts = translator.translate([s["text"] for s in segs])
            _dump(tfile, texts)

        # ---- 5. synthesis ----
        synth = Synthesizer(config.models_dir(), language.voice)
        tts_dir = ld / "tts"

        def synth_line(text):
            return synth.line(text, tts_dir, language.speaker)

        lines = []
        for i, (s, tgt) in enumerate(zip(segs, texts)):
            report("synthesize", 0.45 + 0.35 * i / max(len(segs), 1), f"line {i + 1}/{len(segs)}")
            path, dur = synth_line(tgt)
            lines.append({"start": s["start"], "end": s["end"], "text": s["text"],
                          "text_tgt": tgt, "raw_dur": dur, "wav": str(path)})
        for i, ln in enumerate(lines):
            ln["avail"] = available_time(lines, i)
        baseline = copy.deepcopy(lines)

        outputs = {}
        if use_agent:
            outputs["baseline"] = str(_render(src, info["has_video"], baseline, background, ld,
                                              out_root, stem, lang, "baseline", audio_format))

            # ---- 6. agent ----
            llm = sim_fn = None
            if language.llm_rewrite:
                client = LLM()
                if client.available():
                    llm = client
                    sim_fn = make_similarity(config.models_dir())
                else:
                    warnings.append("Ollama is not running or the model is missing; "
                                    "LLM rewriting was skipped (n-best selection still ran).")
            agent = ShorteningAgent(translator, synth_line, language, llm, sim_fn,
                                    AgentConfig(thr=thr))
            trace = []
            todo = [i for i, ln in enumerate(lines)
                    if classify(ln["raw_dur"], ln["avail"], thr) == "fixable"]
            for k, i in enumerate(todo):
                report("agent", 0.80 + 0.12 * k / max(len(todo), 1), f"fixing line {i + 1}")
                t = agent.fix(lines[i])
                t["line"] = i
                trace.append(t)
        else:
            trace = []

        # ---- 7. render ----
        report("render", 0.95, "mixing final audio")
        tag = "agent" if use_agent else "dubbed"
        outputs[tag] = str(_render(src, info["has_video"], lines, background, ld, out_root,
                                   stem, lang, tag, audio_format))
        summary = summarize(baseline, lines, thr)
        summary.update(language=lang, source=source, seconds=round(time.time() - t0, 1),
                       input=src.name, agent=use_agent)
        result = {"outputs": outputs, "summary": summary, "lines": lines,
                  "baseline_lines": baseline, "trace": trace, "warnings": warnings,
                  "work_dir": str(ld),
                  "context": {"input": str(src), "background": str(background),
                              "has_video": info["has_video"], "stem": stem, "lang": lang, "source": source,
                              "thr": thr, "audio_format": audio_format,
                              "out_dir": str(out_root)}}
        _dump(ld / "report.json", result)
        report("done", 1.0, "finished")
        return result
    finally:
        if synth is not None:
            synth.close()
        if translator is not None:
            translator.close()
        gpu.free()


def redub(result, new_texts, progress=None):
    """Re-render a finished job after the user edited the target-language lines.

    Only lines whose text changed are re-synthesized (the rest come from the TTS cache), so
    this is much faster than a full run.

    Args:
        result: the dict returned by :func:`run` (must contain ``context``).
        new_texts: list of target-language strings, one per line; blank entries keep the
            current text.
        progress: optional callback ``progress(stage, fraction, message)``.

    Returns:
        dict with ``outputs`` (``{"edited": path}``), ``lines`` and ``changed`` (line indices).

    Raises:
        InputError: if ``new_texts`` does not match the number of lines.
    """
    ctx = result["context"]
    lines = copy.deepcopy(result["lines"])
    if len(new_texts) != len(lines):
        raise InputError(f"Expected {len(lines)} lines, got {len(new_texts)}.")
    language = config.get_language(ctx["lang"])
    changed = [i for i, t in enumerate(new_texts)
               if str(t).strip() and str(t).strip() != lines[i]["text_tgt"]]
    synth = None
    try:
        if changed:
            synth = Synthesizer(config.models_dir(), language.voice)
            tts_dir = Path(result["work_dir"]) / "tts"
            for k, i in enumerate(changed):
                if progress:
                    progress("synthesize", k / len(changed), f"line {i + 1}")
                text = str(new_texts[i]).strip()
                path, dur = synth.line(text, tts_dir, language.speaker)
                lines[i].update(text_tgt=text, raw_dur=dur, wav=str(path))
        if progress:
            progress("render", 0.9, "mixing final audio")
        out = _render(ctx["input"], ctx["has_video"], lines, ctx["background"],
                      result["work_dir"], ctx["out_dir"], ctx["stem"], ctx["lang"], "edited",
                      ctx["audio_format"])
    finally:
        if synth is not None:
            synth.close()
        gpu.free()
    return {"outputs": {"edited": str(out)}, "lines": lines, "changed": changed}
