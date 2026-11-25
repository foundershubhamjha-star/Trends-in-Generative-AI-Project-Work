"""Text-to-speech with Indic Parler-TTS, with on-disk caching and runaway protection."""
from pathlib import Path

import soundfile as sf

from . import gpu
from .segments import cache_key, is_runaway


class Synthesizer:
    """Wraps Indic Parler-TTS. One instance holds the model on the GPU (about 4 GB)."""

    def __init__(self, models_dir, voice, folder="indic-parler-tts"):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        path = str(Path(models_dir) / folder)
        self._torch = torch
        self.voice = voice
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(path).to("cuda")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sr = self.model.config.sampling_rate

    def synth(self, text, seed=0):
        """Synthesize ``text`` and return a 1-D float32 array at ``self.sr``."""
        self._torch.manual_seed(seed)
        desc = self.desc_tok(self.voice, return_tensors="pt").to("cuda")
        prompt = self.tok(text, return_tensors="pt").to("cuda")
        with self._torch.no_grad():
            gen = self.model.generate(
                input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask)
        return gen.cpu().numpy().squeeze()

    def line(self, text, cache_dir, voice_key, retries=2):
        """Synthesize one line, cached by content. Returns ``(wav_path, duration_seconds)``.

        If the model produces implausibly long audio (a known failure mode), it retries with a
        different seed and keeps the shortest result.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{cache_key(voice_key, text)}.wav"
        if path.exists():
            return path, sf.info(str(path)).duration
        best = None
        for seed in range(retries + 1):
            y = self.synth(text, seed)
            dur = len(y) / self.sr
            if best is None or dur < best[1]:
                best = (y, dur)
            if not is_runaway(text, dur):
                break
        sf.write(str(path), best[0], self.sr)
        return path, best[1]

    def close(self):
        del self.model
        gpu.free()
