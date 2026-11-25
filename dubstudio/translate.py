"""Translation with IndicTrans2 (distilled 200M), English -> Indic or Indic -> English."""
from pathlib import Path

from . import gpu


class Translator:
    """Wraps IndicTrans2. Loads onto the GPU in half precision (about 0.6 GB)."""

    def __init__(self, models_dir, target_code, folder="indictrans2-en-indic",
                 source_code="eng_Latn"):
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        path = str(Path(models_dir) / folder)
        self._torch = torch
        self.target = target_code
        self.source = source_code
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")
        self.ip = IndicProcessor(inference=True)

    def _generate(self, texts, beams, n_return, n_pre=None):
        """Translate ``texts``; returns ``len(texts) * n_return`` strings.

        ``n_pre`` lets callers pre-process the source several times so that the toolkit's
        per-sentence placeholder bookkeeping stays aligned with ``n_return`` outputs.
        """
        batch = self.ip.preprocess_batch(texts * (n_pre or 1), src_lang=self.source,
                                         tgt_lang=self.target)
        batch = batch[: len(texts)]
        inp = self.tok(batch, truncation=True, padding="longest", return_tensors="pt").to("cuda")
        with self._torch.no_grad():
            out = self.model.generate(**inp, num_beams=beams, num_return_sequences=n_return,
                                      max_length=256, use_cache=False)
        dec = self.tok.batch_decode(out, skip_special_tokens=True)
        return self.ip.postprocess_batch(dec, lang=self.target)

    def translate(self, texts, batch_size=8):
        """Translate a list of source-language sentences (best beam only)."""
        result = []
        for i in range(0, len(texts), batch_size):
            result += self._generate(list(texts[i:i + batch_size]), beams=5, n_return=1)
        return result

    def nbest(self, text, n=6, beams=8):
        """Return up to ``n`` alternative translations of one sentence, best first."""
        cands = self._generate([text], beams=max(beams, n), n_return=n, n_pre=n)
        seen, unique = set(), []
        for c in cands:
            if c and c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def close(self):
        del self.model
        gpu.free()
