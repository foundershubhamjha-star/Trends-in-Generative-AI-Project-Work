"""The shortening agent.

When a dubbed line is too long for its time slot, the agent escalates through strategies,
cheapest first, and verifies every candidate by actually synthesizing it:

1. **n-best selection** (no LLM): IndicTrans2 proposes several translations of the same
   English sentence; shorter ones are tried. Every candidate is a direct translation, so
   meaning is preserved by construction. Works for every supported language.
2. **LLM rewrite** (only for languages marked ``llm_rewrite``): a local LLM shortens the line.
   A rewrite is accepted only if it passes (a) a length floor, (b) a back-translation
   similarity check, and (c) an LLM yes/no check that key facts survive. Rejected attempts
   are fed back into the next prompt together with the reason (Reflexion-style), and the
   temperature rises on each retry.

The agent never accepts a candidate that is not strictly shorter in synthesized duration than
the current best, so it cannot make a line worse.
"""
import logging
from dataclasses import dataclass

from .config import DEFAULT_THR

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Tunable behaviour of the agent."""
    thr: float = DEFAULT_THR   # accepted ratio of speech duration to available time
    tries: int = 4             # LLM rewrite attempts per line
    sim_min: float = 0.70      # minimum back-translation similarity
    min_ratio: float = 0.5     # a rewrite may not be shorter than this fraction of the original
    nbest_n: int = 6           # candidates requested from the translator
    nbest_tries: int = 3       # candidates actually synthesized


class ShorteningAgent:
    """Fixes lines whose speech is too long for their slot.

    Args:
        translator: object with ``nbest(text, n) -> list[str]``.
        synth_line: callable ``text -> (wav_path, duration_seconds)``.
        lang: a :class:`dubstudio.config.Language`.
        llm: optional object with ``chat(prompt, temperature) -> str``.
        sim_fn: optional callable ``(english_a, english_b) -> float`` (needed with ``llm``).
        config: an :class:`AgentConfig`.
    """

    def __init__(self, translator, synth_line, lang, llm=None, sim_fn=None, config=None):
        self.translator = translator
        self.synth_line = synth_line
        self.lang = lang
        self.llm = llm
        self.sim_fn = sim_fn
        self.cfg = config or AgentConfig()

    # ---------------------------------------------------------------- public API
    def fix(self, seg):
        """Try to make ``seg`` fit. Updates ``text_tgt``, ``raw_dur`` and ``wav`` in place.

        ``seg`` needs: ``text`` (English), ``text_tgt``, ``avail``, ``raw_dur``, ``wav``.
        Returns a trace dict describing every step taken.
        """
        limit = seg["avail"] * self.cfg.thr
        original = seg["text_tgt"]
        best = {"text": original, "dur": seg["raw_dur"], "path": seg["wav"]}
        trace = {"english": seg["text"], "before_dur": round(seg["raw_dur"], 2),
                 "avail": round(seg["avail"], 2), "steps": []}

        self._nbest(seg, original, best, limit, trace["steps"])
        if best["dur"] > limit and self.llm is not None and self.lang.llm_rewrite:
            try:
                self._llm_rewrite(seg, original, best, limit, trace["steps"])
            except Exception as exc:  # network/LLM failure must not abort the whole job
                log.warning("LLM rewrite failed: %s", exc)
                trace["steps"].append({"strategy": "llm", "error": str(exc)})

        seg["text_tgt"], seg["raw_dur"], seg["wav"] = best["text"], best["dur"], str(best["path"])
        trace["after_dur"] = round(best["dur"], 2)
        trace["fixed"] = best["dur"] <= limit
        return trace

    # ------------------------------------------------------------ strategy 1: n-best
    def _nbest(self, seg, original, best, limit, steps):
        cfg = self.cfg
        try:
            cands = self.translator.nbest(seg["text"], n=cfg.nbest_n)
        except Exception as exc:
            log.warning("n-best generation failed: %s", exc)
            steps.append({"strategy": "nbest", "error": str(exc)})
            return
        cands = [c for c in cands
                 if len(c) < len(original) and len(c) >= cfg.min_ratio * len(original)]

        # Prefer the longest candidate predicted to fit (keeps the most content), then the
        # shortest as a fallback.
        def predicted(c):
            return seg["raw_dur"] * len(c) / max(len(original), 1)

        fits = sorted((c for c in cands if predicted(c) <= limit), key=len, reverse=True)
        rest = sorted((c for c in cands if predicted(c) > limit), key=len)
        order = (fits[:2] + rest[:1])[: cfg.nbest_tries]
        for cand in order:
            path, dur = self.synth_line(cand)
            steps.append({"strategy": "nbest", "chars": len(cand), "raw_dur": round(dur, 2),
                          "text": cand, "ok_time": dur <= limit})
            if dur < best["dur"]:
                best.update(text=cand, dur=dur, path=path)
            if dur <= limit:
                return

    # ------------------------------------------------------------ strategy 2: LLM
    def _judge(self, english, long_line, short_line):
        reply = self.llm.chat(
            f"You compare two {self.lang.name} dubbing lines for the same original sentence.\n"
            f"Original: {english}\nLong: {long_line}\nShort: {short_line}\n"
            "Does the Short line keep every key fact, name and list item of the original sentence "
            "(wording may be shorter)? Answer with exactly YES or NO.", temperature=0.0)
        return reply.strip().upper().startswith("YES")

    def _llm_rewrite(self, seg, original, best, limit, steps):
        cfg = self.cfg
        cur = best["text"]
        feedback = ""
        for t in range(cfg.tries):
            ratio = seg["avail"] / max(best["dur"], 0.1)
            target = max(int(len(cur) * min(0.95, ratio * 0.95)), int(len(original) * cfg.min_ratio))
            prompt = (f"Rewrite this {self.lang.name} dubbing line so it is shorter, about {target} "
                      f"characters (currently {len(cur)}). Keep every key fact, name and list item. "
                      f"Output only the {self.lang.name} sentence.")
            if feedback:
                prompt += "\n" + feedback
            cand = self.llm.chat(prompt + "\n" + cur, temperature=min(0.2 + 0.25 * t, 0.9))
            step = {"strategy": "llm", "try": t, "chars": len(cand)}
            if not cand or len(cand) >= len(cur):
                step["rejected"] = "not shorter"
                feedback = "Your previous attempt was not shorter. Make it clearly shorter."
                steps.append(step)
                continue
            if len(cand) < cfg.min_ratio * len(original):
                step["rejected"] = "too short"
                feedback = "Your previous attempt dropped too much content. Keep all key facts."
                steps.append(step)
                continue
            if self.lang.code == "en":
                back = cand   # the multilingual embedding compares English with the source directly
            else:
                back = self.llm.chat(f"Translate this {self.lang.name} sentence into English. "
                                     f"Output only the English.\n{cand}")
            sim = self.sim_fn(seg["text"], back)
            ok_sim = sim >= cfg.sim_min
            ok_judge = self._judge(seg["text"], original, cand) if ok_sim else False
            ok = ok_sim and ok_judge
            path, dur = self.synth_line(cand)
            step.update(sim=round(sim, 3), judge=ok_judge, raw_dur=round(dur, 2),
                        ok_meaning=ok, ok_time=dur <= limit, text=cand, back_en=back)
            steps.append(step)
            if ok and dur < best["dur"]:
                best.update(text=cand, dur=dur, path=path)
            if ok and dur <= limit:
                return
            if ok:
                feedback = "Your previous attempt kept the meaning but was still too long. Shorten more."
                cur = cand
            else:
                feedback = ("Your previous attempt changed the meaning or lost facts. "
                            "Shorten by removing filler words, not content.")


def make_similarity(models_dir, folder="minilm"):
    """Build a sentence-similarity function from a multilingual MiniLM model (CPU).

    The model is cross-lingual, so it can compare a Hindi sentence with an English one.
    """
    from pathlib import Path

    from sentence_transformers import SentenceTransformer, util

    emb = SentenceTransformer(str(Path(models_dir) / folder), device="cpu")

    def sim(a, b):
        return float(util.cos_sim(emb.encode(a), emb.encode(b)))

    return sim
