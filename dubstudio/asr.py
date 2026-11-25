"""Speech recognition with faster-whisper."""
from pathlib import Path

from . import audio, gpu


def transcribe(vocals_path, model_dir, tmp_dir):
    """Transcribe English speech.

    Returns ``(segments, language, language_probability)`` where ``segments`` is a list of
    ``{"start", "end", "text"}``. The language is auto-detected so callers can refuse
    non-English input instead of producing nonsense.
    """
    gpu.prepare_cuda_dlls()
    from faster_whisper import WhisperModel

    tmp = Path(tmp_dir) / "vocals16k.wav"
    audio.to_16k_mono(vocals_path, tmp)
    model = WhisperModel(str(model_dir), device="cuda", compute_type="int8_float16")
    try:
        seg_iter, info = model.transcribe(str(tmp), language=None, vad_filter=True, beam_size=5)
        segments = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                    for s in seg_iter if s.text.strip()]
    finally:
        del model
        gpu.free()
    return segments, info.language, float(info.language_probability)
