"""Agent layer on top of the baseline dubbing pipeline.

For every line whose Hindi speech needs more than --thr x speed-up to fit its slot:
  1. shorten the Hindi line with an LLM (Ollama),
  2. translate it back to English and compare with the source (meaning check),
  3. synthesize it and check the raw duration against the slot,
  4. retry up to --tries times and keep the best verified candidate.

With --mem 1 it also uses (a) a glossary for consistent term translations and
(b) rewrite memory: earlier successful shortenings are retrieved as few-shot examples.

Prerequisites: python dub.py <input> has been run, and work/<name>/report.json has been
copied to work/<name>/report_baseline.json. Ollama must be running.

Usage:
    python agent2.py samples/clip.wav --mem 0 --thr 1.15
    python agent2.py samples/clip.wav --mem 1 --thr 1.15
"""
import os
import re
import json
import gc
import argparse

import requests
import soundfile as sf
import torch

import dub

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"
MEM_PATH = "work/memory.json"
MIN_SLOT = 0.6  # segments with a shorter slot are too short to rewrite


def llm(prompt):
    r = requests.post(OLLAMA, json={
        "model": MODEL, "stream": False, "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": 400},
        "messages": [
            {"role": "system",
             "content": "You are a precise Hindi dubbing editor. Output only what is asked, no explanations."},
            {"role": "user", "content": prompt}]}, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip().strip('"').strip()


def mem_load():
    if os.path.exists(MEM_PATH):
        return json.load(open(MEM_PATH, encoding="utf-8"))
    return {"glossary": {}, "rewrites": []}


def mem_save(m):
    m["rewrites"] = m["rewrites"][-200:]
    json.dump(m, open(MEM_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def parse_list(t):
    m = re.search(r"\[.*?\]", t, re.S)
    if not m:
        return []
    try:
        return [x.strip() for x in json.loads(m.group(0)) if isinstance(x, str) and x.strip()]
    except Exception:
        return []


def synth(model, tok, dtok, text, seed=0):
    torch.manual_seed(seed)
    dd = dtok(dub.VOICE, return_tensors="pt").to("cuda")
    pp = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        g = model.generate(input_ids=dd.input_ids, attention_mask=dd.attention_mask,
                           prompt_input_ids=pp.input_ids, prompt_attention_mask=pp.attention_mask)
    return g.cpu().numpy().squeeze()


def fit(y, sr, avail, out):
    """Time-fit audio y into `avail` seconds. Returns (fitted_duration, raw_duration)."""
    dur = len(y) / sr
    tempo = min(dur / avail, dub.MAX_TEMPO) if dur > avail else 1.0
    raw = out + ".raw.wav"
    sf.write(raw, y, sr)
    if tempo > 1.01:
        dub.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                 "-filter:a", f"atempo={tempo:.3f}", out])
    else:
        sf.write(out, y, sr)
    os.remove(raw)
    y2, _ = sf.read(out)
    return len(y2) / sr, dur


def shots(mem, emb, util, en, k=3):
    """Retrieve the k most similar earlier rewrites as few-shot examples."""
    rs = mem["rewrites"]
    if not rs:
        return ""
    sims = util.cos_sim(emb.encode(en), emb.encode([r["en"] for r in rs]))[0].tolist()
    top = sorted(range(len(rs)), key=lambda i: -sims[i])[:k]
    return "\n".join(f"Long: {rs[i]['before']}\nShort: {rs[i]['after']}" for i in top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--mem", type=int, default=0, help="1 = use glossary + rewrite memory")
    ap.add_argument("--thr", type=float, default=1.15,
                    help="allowed ratio of raw speech duration to slot")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--sim", type=float, default=0.70,
                    help="minimum back-translation similarity to accept a rewrite")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.video))[0]
    wd = f"work/{name}"
    mode = "agentmem" if args.mem else "agent"
    md = f"{wd}/{mode}"
    os.makedirs(f"{md}/tts", exist_ok=True)
    segs = json.load(open(f"{wd}/report_baseline.json", encoding="utf-8"))
    is_audio = os.path.splitext(args.video)[1].lower() in dub.AUDIO_EXTS
    outp = f"{name}_hi_{mode}." + ("wav" if is_audio else "mp4")

    from sentence_transformers import SentenceTransformer, util
    emb = SentenceTransformer(r".\models\minilm", device="cpu")
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    path = r".\models\indic-parler-tts"
    model = ParlerTTSForConditionalGeneration.from_pretrained(path).to("cuda")
    tok = AutoTokenizer.from_pretrained(path)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate

    # Rebuild baseline audio in this mode's own folder; measure RAW duration (before speed-up).
    for i, s in enumerate(segs):
        y, _ = sf.read(f"{wd}/tts/{i:04d}_raw.wav")
        _, rd = fit(y, sr, s["avail"], f"{md}/tts/{i:04d}.wav")
        s["tts_dur"] = round(rd, 2)
        s["overshoot"] = bool(rd > s["avail"] * args.thr)

    mem = mem_load()
    gloss_edits = 0
    seg_terms = {}

    # ---- memory part 1: glossary for consistent terminology ----
    if args.mem:
        text = " ".join(s["text"] for s in segs)[:4000]
        terms = parse_list(llm(
            "List up to 12 proper nouns or technical terms from this transcript that must be "
            "translated consistently. Output only a JSON list of strings.\n" + text))
        for t in terms:
            if t.lower() not in mem["glossary"]:
                hi = llm(f"Give the standard Hindi (Devanagari) rendering of the term '{t}' as used "
                         f"in a talk. If it is normally kept in English, give its Devanagari "
                         f"transliteration. Output only the Hindi word or phrase.")
                hi = hi.splitlines()[0].strip() if hi else ""
                # accept only real Devanagari renderings (no commentary, no English echoes)
                if hi and len(hi) < 60 and re.search("[\u0900-\u097F]", hi) and "(" not in hi:
                    mem["glossary"][t.lower()] = hi
        for i, s in enumerate(segs):
            seg_terms[i] = {t: hi for t, hi in mem["glossary"].items() if t in s["text"].lower()}
            for t, hi in seg_terms[i].items():
                if hi in s["text_hi"]:
                    continue
                new = llm(f"Edit this Hindi sentence minimally so the term '{t}' is rendered as "
                          f"'{hi}'. Keep everything else the same. Output only the Hindi sentence.\n"
                          f"English: {s['text']}\nHindi: {s['text_hi']}")
                if hi in new and 0.7 <= len(new) / max(len(s["text_hi"]), 1) <= 1.4:
                    s["text_hi"] = new
                    y = synth(model, tok, dtok, new, seed=0)
                    _, rd = fit(y, sr, s["avail"], f"{md}/tts/{i:04d}.wav")
                    s["tts_dur"] = round(rd, 2)
                    s["overshoot"] = bool(rd > s["avail"] * args.thr)
                    gloss_edits += 1

    # ---- agent loop: shorten, verify meaning, verify timing, retry ----
    trace = []
    for i, s in enumerate(segs):
        if not s["overshoot"] or s["avail"] < MIN_SLOT:
            continue
        src_en = s["text"]
        orig_hi = s["text_hi"]
        best = {"text": s["text_hi"], "dur": s["tts_dur"]}
        log = {"seg": i, "english": src_en, "before_dur": s["tts_dur"],
               "avail": s["avail"], "attempts": []}
        cur = s["text_hi"]
        for t in range(args.tries):
            ratio = s["avail"] / max(best["dur"], 0.1)
            target = int(len(cur) * min(0.95, ratio * 0.95))
            p = (f"Rewrite this Hindi dubbing line so it is shorter, about {target} characters "
                 f"(currently {len(cur)}). Keep the meaning and key terms. "
                 f"Output only the Hindi sentence.")
            if args.mem:
                ex = shots(mem, emb, util, src_en)
                if ex:
                    p = "Examples of good shortening:\n" + ex + "\n\n" + p
                if seg_terms.get(i):
                    p += "\nUse these term renderings: " + "; ".join(
                        f"{a} = {b}" for a, b in seg_terms[i].items())
            cand = llm(p + "\n" + cur)
            # guard: a rewrite must be shorter than the current line, but not gutted
            if len(cand) >= len(cur) or len(cand) < 0.4 * len(cur):
                log["attempts"].append({"try": t, "chars": len(cand), "rejected": "bad length"})
                continue
            back = llm("Translate this Hindi sentence into English. Output only the English.\n" + cand)
            sim = float(util.cos_sim(emb.encode(src_en), emb.encode(back)))
            ok_m = sim >= args.sim
            y = synth(model, tok, dtok, cand, seed=t)
            fitted, rawd = fit(y, sr, s["avail"], f"{md}/tts/{i:04d}.cand.wav")
            ok_t = rawd <= s["avail"] * args.thr
            log["attempts"].append({"try": t, "chars": len(cand), "sim": round(sim, 3),
                                    "fitted_dur": round(fitted, 2), "raw_dur": round(rawd, 2),
                                    "ok_meaning": ok_m, "ok_time": ok_t,
                                    "text_hi": cand, "back_en": back})
            if ok_m and rawd < best["dur"]:
                best = {"text": cand, "dur": rawd}
                os.replace(f"{md}/tts/{i:04d}.cand.wav", f"{md}/tts/{i:04d}.wav")
            if ok_m and ok_t:
                if args.mem:
                    mem["rewrites"].append({"en": src_en, "before": orig_hi, "after": cand,
                                            "ratio": round(len(cand) / max(len(orig_hi), 1), 2)})
                break
            cur = cand if ok_m else cur
        s["text_hi"] = best["text"]
        s["tts_dur"] = round(best["dur"], 2)
        s["overshoot"] = bool(best["dur"] > s["avail"] * args.thr)
        log["after_dur"] = s["tts_dur"]
        log["fixed"] = not s["overshoot"]
        trace.append(log)
        cf = f"{md}/tts/{i:04d}.cand.wav"
        if os.path.exists(cf):
            os.remove(cf)

    if args.mem:
        mem_save(mem)
    json.dump(segs, open(f"{md}/report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(trace, open(f"{md}/trace.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    bg = f"{wd}/sep/htdemucs/audio/no_vocals.wav"
    dub.stage_assemble(args.video, segs, bg, sr, md, outp)
    n_fix = sum(1 for t in trace if t["fixed"])
    n_left = sum(1 for s in segs if s["overshoot"] and s["avail"] >= MIN_SLOT)
    summ = {"mode": mode, "segments": len(segs), "flagged": len(trace), "fixed": n_fix,
            "overshoot_left": n_left, "glossary_edits": gloss_edits}
    json.dump(summ, open(f"{md}/summary.json", "w"), indent=1)
    print("DONE ->", outp, "|", summ)


if __name__ == "__main__":
    main()
