"""Compare an original English transcript with the English that comes back after a round trip.

Round trip: English video -> Hindi dub -> (Hindi -> English dub). Run the three steps, then:

    python tools/roundtrip.py --original jfk_english --dubbed jfk_english_hi_agent

``--original`` is the work-folder name of the first English input, ``--dubbed`` the name of the
Hindi dub that was dubbed back. The script prints both texts and two similarity scores.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def words(text):
    """Lower-case word list without punctuation."""
    return re.findall(r"[a-z0-9']+", text.lower())


def rouge1_f1(reference, candidate):
    """Unigram F1 between two texts (1.0 = same words, 0.0 = nothing in common)."""
    ref, cand = Counter(words(reference)), Counter(words(candidate))
    overlap = sum((ref & cand).values())
    if not overlap:
        return 0.0
    p, r = overlap / sum(cand.values()), overlap / sum(ref.values())
    return 2 * p * r / (p + r)


def load_original(work, name):
    """Full English transcript from the first run's ASR output."""
    data = json.loads((Path(work) / name / "asr.json").read_text(encoding="utf-8"))
    return " ".join(s["text"] for s in data["segments"])


def load_roundtrip(work, name, target="en", source="hi"):
    """English text produced by the Hindi -> English run."""
    rep = json.loads((Path(work) / name / f"{source}-{target}" / "report.json").read_text(encoding="utf-8"))
    return " ".join(l["text_tgt"] for l in rep["lines"])


def embedding_similarity(a, b, models_dir="models"):
    """Cosine similarity of the two texts with the local MiniLM model, or None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer(str(Path(models_dir) / "minilm"), device="cpu")
        return float(util.cos_sim(model.encode(a), model.encode(b)))
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True)
    ap.add_argument("--dubbed", required=True)
    ap.add_argument("--work", default="work")
    args = ap.parse_args()
    orig = load_original(args.work, args.original)
    back = load_roundtrip(args.work, args.dubbed)
    print("ORIGINAL ENGLISH:\n" + orig + "\n")
    print("AFTER ROUND TRIP:\n" + back + "\n")
    print(f"word overlap (ROUGE-1 F1): {rouge1_f1(orig, back):.2f}")
    sim = embedding_similarity(orig, back)
    if sim is not None:
        print(f"meaning similarity (embedding cosine): {sim:.2f}")
    print("\nTwo translations in a row never return the exact words; judge by meaning.")


if __name__ == "__main__":
    raise SystemExit(main())
