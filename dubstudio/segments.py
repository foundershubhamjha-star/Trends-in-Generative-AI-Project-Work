"""Pure logic on transcript segments: merging, timing, classification, summaries.

No heavy imports here, so everything in this module is unit-tested without a GPU.
A *segment* is a dict with at least ``start``, ``end`` (seconds) and ``text``.
"""
import hashlib
from collections import Counter

from .config import MAX_RATIO, MAX_TEMPO, MIN_SLOT

SENTENCE_END = (".", "?", "!", "।", '"', "”", "’")


def ends_sentence(text: str) -> bool:
    """True if ``text`` ends with sentence-final punctuation."""
    return text.rstrip().endswith(SENTENCE_END)


def merge_segments(segs, min_dur=1.5, max_dur=12.0, max_gap=0.8):
    """Merge fragments so each dubbed line is a sentence-sized unit with a usable time slot.

    Speech recognition often cuts mid-sentence or emits sub-second fragments. Such fragments
    translate badly and leave no time to speak the translation. Two neighbours are merged when
    they are close together (gap <= ``max_gap``), the result is not too long (<= ``max_dur``),
    and at least one of: the first piece is shorter than ``min_dur``, the second piece is
    shorter than ``min_dur``, or the first piece does not end a sentence.

    Returns new dicts; the input is not modified.
    """
    out = []
    for seg in segs:
        seg = dict(seg)
        if out:
            cur = out[-1]
            gap = seg["start"] - cur["end"]
            total = seg["end"] - cur["start"]
            wants = ((cur["end"] - cur["start"]) < min_dur
                     or (seg["end"] - seg["start"]) < min_dur
                     or not ends_sentence(cur["text"]))
            if wants and gap <= max_gap and total <= max_dur:
                cur["end"] = seg["end"]
                cur["text"] = (cur["text"].rstrip() + " " + seg["text"].lstrip()).strip()
                continue
        out.append(seg)
    return out


def available_time(segs, i, borrow=1.3):
    """Seconds line ``i`` may occupy: its own slot, extended into the gap before the next line.

    The extension is capped at ``borrow`` times the slot and never runs into the next line.
    """
    seg = segs[i]
    slot = seg["end"] - seg["start"]
    nxt = segs[i + 1]["start"] if i + 1 < len(segs) else seg["end"] + 2.0
    return max(slot, min(nxt - seg["start"] - 0.05, slot * borrow))


def tempo_for(dur, avail, max_tempo=MAX_TEMPO):
    """Speed-up factor needed to fit ``dur`` seconds of speech into ``avail`` seconds."""
    if avail <= 0 or dur <= avail:
        return 1.0
    return min(dur / avail, max_tempo)


def is_runaway(text, dur):
    """True if synthesized audio is implausibly long for its text (a TTS failure mode)."""
    return dur > len(text) / 6 + 1.5


def cache_key(*parts) -> str:
    """Short stable hash used to name cache files by content."""
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:16]


def classify(raw_dur, avail, thr):
    """Classify a line as ``ok``, ``fixable`` or ``artifact``.

    ``artifact`` means the slot is too short to hold any line (below ``MIN_SLOT``) or the
    speech is more than ``MAX_RATIO`` times longer than the slot, which points to a
    transcript timing problem and not something shortening the text can solve.
    """
    if raw_dur <= avail * thr:
        return "ok"
    if avail < MIN_SLOT or raw_dur / max(avail, 1e-6) > MAX_RATIO:
        return "artifact"
    return "fixable"


def summarize(before, after, thr):
    """Compare lines before and after the agent. Both lists are index-aligned."""
    kinds_before = [classify(b["raw_dur"], b["avail"], thr) for b in before]
    kinds_after = [classify(a["raw_dur"], a["avail"], thr) for a in after]
    counts = Counter(kinds_before)
    fixed = sum(1 for kb, ka in zip(kinds_before, kinds_after) if kb == "fixable" and ka == "ok")
    tb = [tempo_for(b["raw_dur"], b["avail"]) for b in before]
    ta = [tempo_for(a["raw_dur"], a["avail"]) for a in after]

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "lines": len(before),
        "fixable_before": counts.get("fixable", 0),
        "fixed": fixed,
        "fixable_after": kinds_after.count("fixable"),
        "artifacts": counts.get("artifact", 0),
        "mean_tempo_before": mean(tb),
        "mean_tempo_after": mean(ta),
        "max_tempo_before": round(max(tb), 3) if tb else 0.0,
        "max_tempo_after": round(max(ta), 3) if ta else 0.0,
    }
