"""Audio/video helpers built on ffmpeg, ffprobe, Demucs and soundfile."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import DubStudioError, InputError


def run(cmd):
    """Run an external command; raise :class:`DubStudioError` with its stderr on failure."""
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        name = Path(str(cmd[0])).name
        raise DubStudioError(f"{name} failed: {proc.stderr.strip()[-400:]}")
    return proc.stdout


def probe(path):
    """Return ``{"duration", "has_audio", "has_video"}`` for a media file.

    Cover art embedded in an MP3 is not counted as video.
    """
    try:
        out = run(["ffprobe", "-v", "error", "-show_entries",
                   "stream=codec_type:stream_disposition=attached_pic:format=duration",
                   "-of", "json", path])
        info = json.loads(out)
    except (DubStudioError, json.JSONDecodeError):
        raise InputError("Could not read this file as audio or video.") from None
    streams = info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_video = any(s.get("codec_type") == "video"
                    and not s.get("disposition", {}).get("attached_pic", 0) for s in streams)
    try:
        duration = float(info.get("format", {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    return {"duration": duration, "has_audio": has_audio, "has_video": has_video}


def extract_audio(src, dst):
    """Extract/convert any input to 44.1 kHz stereo WAV."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "2", "-ar", "44100", dst])


def to_16k_mono(src, dst):
    """Convert to 16 kHz mono WAV, the format speech recognition expects."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", dst])


def separate_vocals(audio_path, out_dir):
    """Split speech from background with Demucs. Returns ``(vocals_path, background_path)``."""
    audio_path, out_dir = Path(audio_path), Path(out_dir)
    stem = audio_path.stem
    vocals = out_dir / "htdemucs" / stem / "vocals.wav"
    background = out_dir / "htdemucs" / stem / "no_vocals.wav"
    if not vocals.exists() or not background.exists():
        run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-d", "cuda",
             "-o", out_dir, audio_path])
    if not vocals.exists() or not background.exists():
        raise DubStudioError("Vocal separation did not produce the expected files.")
    return vocals, background


def speed_up(src, dst, tempo):
    """Time-compress ``src`` into ``dst`` by ``tempo`` (pitch preserved). Copies if ~1.0."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if tempo <= 1.01:
        shutil.copyfile(src, dst)
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-filter:a", f"atempo={tempo:.3f}", dst])


def mix(background_path, clips, out_path, bg_gain=0.9):
    """Place ``clips`` (``[(start_seconds, wav_path), ...]``) over the background track.

    Returns the sample rate. The result is peak-limited to avoid clipping.
    """
    bg, sr = sf.read(str(background_path), dtype="float32")
    out = bg * bg_gain
    for start, path in clips:
        y, ysr = sf.read(str(path), dtype="float32")
        if ysr != sr:
            raise DubStudioError(f"Sample-rate mismatch ({ysr} vs {sr}).")
        if y.ndim == 2:
            y = y.mean(axis=1)
        a = int(start * sr)
        if a >= len(out):
            continue
        b = min(a + len(y), len(out))
        seg = y[: b - a]
        if out.ndim == 2:
            seg = np.stack([seg, seg], axis=1)
        out[a:b] += seg
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0.98:
        out = out / peak * 0.98
    sf.write(str(out_path), out, sr)
    return sr


def mux(video_path, audio_path, out_path):
    """Replace the audio track of ``video_path`` with ``audio_path``."""
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-i", audio_path,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", out_path])


def export_audio(wav_path, out_path, fmt="wav"):
    """Write the final audio as WAV (copy) or MP3."""
    if fmt == "mp3":
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
             "-codec:a", "libmp3lame", "-q:a", "2", out_path])
    else:
        shutil.copyfile(wav_path, out_path)
