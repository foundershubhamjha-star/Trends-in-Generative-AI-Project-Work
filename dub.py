"""Baseline English -> Hindi dubbing pipeline.

audio/video -> extract audio -> separate vocals/background (Demucs)
-> ASR with timestamps (faster-whisper) -> translate (IndicTrans2)
-> synthesize + time-fit (Indic Parler-TTS, ffmpeg atempo) -> remix.

Every stage caches its output under work/<name>/, so reruns skip finished work.

Usage:
    python dub.py samples/clip.wav
    python dub.py samples/clip.mp4
"""
import os
import sys
import json
import gc
import shutil
import subprocess
import argparse

import numpy as np
import soundfile as sf
import torch

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac")

# Voice description for Indic Parler-TTS (one voice for all speakers for now).
VOICE = ("Divya's voice is monotone yet slightly fast in delivery, with a very close "
         "recording that almost has no background noise.")

MAX_TEMPO = 1.35  # maximum speed-up applied to fit a line into its slot


def run(cmd):
    subprocess.run(cmd, check=True)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def jsave(p, o):
    json.dump(o, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---------- stage 1: audio extraction ----------
def stage_audio(video, wd):
    a = f"{wd}/audio.wav"
    if not os.path.exists(a):
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
             "-vn", "-ac", "2", "-ar", "44100", a])
    return a


# ---------- stage 2: vocal / background separation ----------
def stage_sep(a, wd):
    out = f"{wd}/sep"
    voc = f"{out}/htdemucs/audio/vocals.wav"
    bg = f"{out}/htdemucs/audio/no_vocals.wav"
    if not os.path.exists(voc):
        run([sys.executable, "-m", "demucs", "--two-stems=vocals",
             "-d", "cuda", "-o", out, a])
    return voc, bg


# ---------- stage 3: speech recognition ----------
def stage_asr(voc, wd):
    p = f"{wd}/segments_en.json"
    if os.path.exists(p):
        return jload(p)
    t16 = f"{wd}/vocals16k.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", voc,
         "-ac", "1", "-ar", "16000", t16])
    from faster_whisper import WhisperModel
    m = WhisperModel(r".\models\faster-whisper-large-v3",
                     device="cuda", compute_type="int8_float16")
    segs, info = m.transcribe(t16, language="en", vad_filter=True, beam_size=5)
    out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
           for s in segs if s.text.strip()]
    del m
    free()
    jsave(p, out)
    return out


# ---------- stage 4: translation ----------
def stage_translate(segs, wd):
    p = f"{wd}/segments_hi.json"
    if os.path.exists(p):
        return jload(p)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor
    path = r".\models\indictrans2-en-indic"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")
    ip = IndicProcessor(inference=True)
    texts = [s["text"] for s in segs]
    res = []
    for i in range(0, len(texts), 8):
        batch = ip.preprocess_batch(texts[i:i + 8], src_lang="eng_Latn", tgt_lang="hin_Deva")
        inp = tok(batch, truncation=True, padding="longest", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, num_beams=5, max_length=256, use_cache=False)
        dec = tok.batch_decode(out, skip_special_tokens=True)
        res += ip.postprocess_batch(dec, lang="hin_Deva")
    for s, h in zip(segs, res):
        s["text_hi"] = h
    del model
    free()
    jsave(p, segs)
    return segs


# ---------- stage 5: speech synthesis + timing fit ----------
def stage_tts(segs, wd):
    d = f"{wd}/tts"
    os.makedirs(d, exist_ok=True)
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    path = r".\models\indic-parler-tts"
    model = ParlerTTSForConditionalGeneration.from_pretrained(path).to("cuda")
    tok = AutoTokenizer.from_pretrained(path)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate
    for i, s in enumerate(segs):
        raw, fit = f"{d}/{i:04d}_raw.wav", f"{d}/{i:04d}.wav"
        if not os.path.exists(raw):
            torch.manual_seed(0)
            dd = dtok(VOICE, return_tensors="pt").to("cuda")
            pp = tok(s["text_hi"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(input_ids=dd.input_ids, attention_mask=dd.attention_mask,
                                   prompt_input_ids=pp.input_ids,
                                   prompt_attention_mask=pp.attention_mask)
            sf.write(raw, g.cpu().numpy().squeeze(), sr)
        y, _ = sf.read(raw)
        dur = len(y) / sr
        slot = s["end"] - s["start"]
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else s["end"] + 2.0
        # a line may borrow time from the gap before the next line (up to 1.3x its slot)
        avail = max(slot, min(nxt - s["start"] - 0.05, slot * 1.3))
        tempo = min(dur / avail, MAX_TEMPO) if dur > avail else 1.0
        if tempo > 1.01:
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                 "-filter:a", f"atempo={tempo:.3f}", fit])
        else:
            sf.write(fit, y, sr)
        y2, _ = sf.read(fit)
        s["tts_dur"] = round(len(y2) / sr, 2)
        s["avail"] = round(avail, 2)
        s["overshoot"] = bool(s["tts_dur"] > avail * 1.1)
    del model
    free()
    return segs, sr


# ---------- stage 6: assemble and mux ----------
def stage_assemble(video, segs, bg_path, sr, wd, outp):
    bg, bsr = sf.read(bg_path)
    assert bsr == sr, f"sample rate mismatch {bsr} vs {sr}"
    mix = bg * 0.9
    for i, s in enumerate(segs):
        y, _ = sf.read(f"{wd}/tts/{i:04d}.wav")
        a = int(s["start"] * sr)
        b = min(a + len(y), len(mix))
        if a >= len(mix):
            continue
        seg = y[: b - a]
        if mix.ndim == 2:
            seg = np.stack([seg, seg], axis=1)
        mix[a:b] += seg
    peak = np.abs(mix).max()
    if peak > 0.98:
        mix = mix / peak * 0.98
    mixp = f"{wd}/dubbed_audio.wav"
    sf.write(mixp, mix, sr)
    if os.path.splitext(video)[1].lower() in AUDIO_EXTS:
        shutil.copy(mixp, outp)  # audio-only input: output is the dubbed audio
        return
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", mixp,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
         "-shortest", outp])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="input audio or video file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    name = os.path.splitext(os.path.basename(args.video))[0]
    is_audio = os.path.splitext(args.video)[1].lower() in AUDIO_EXTS
    wd = f"work/{name}"
    os.makedirs(wd, exist_ok=True)
    outp = args.out or (f"{name}_hi.wav" if is_audio else f"{name}_hi.mp4")

    a = stage_audio(args.video, wd); print("1 audio ok")
    voc, bg = stage_sep(a, wd); print("2 separation ok")
    segs = stage_asr(voc, wd); print("3 ASR ok:", len(segs), "segments")
    segs = stage_translate(segs, wd); print("4 translation ok")
    segs, sr = stage_tts(segs, wd); print("5 TTS ok")
    jsave(f"{wd}/report.json", segs)
    stage_assemble(args.video, segs, bg, sr, wd, outp)
    n_over = sum(s["overshoot"] for s in segs)
    print(f"DONE -> {outp} | segments: {len(segs)} | overshoot: {n_over}")
