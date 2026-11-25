import os, sys, json, gc, argparse, requests, torch, soundfile as sf, numpy as np
import dub

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"

def llm(prompt):
    r = requests.post(OLLAMA, json={
        "model": MODEL, "stream": False, "think": False,
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": "You are a precise Hindi dubbing editor. Output only what is asked, no explanations."},
            {"role": "user", "content": prompt + " /no_think"}]}, timeout=300)
    r.raise_for_status()
    t = r.json()["message"]["content"].strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    return t.strip().strip('"').strip()

def synth(model, tok, dtok, text, seed=0):
    torch.manual_seed(seed)
    dd = dtok(dub.VOICE, return_tensors="pt").to("cuda")
    pp = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        g = model.generate(input_ids=dd.input_ids, attention_mask=dd.attention_mask,
                           prompt_input_ids=pp.input_ids, prompt_attention_mask=pp.attention_mask)
    return g.cpu().numpy().squeeze()

def fit(y, sr, avail, out):
    dur = len(y) / sr
    tempo = min(dur / avail, 1.35) if dur > avail else 1.0
    raw = out + ".raw.wav"
    sf.write(raw, y, sr)
    if tempo > 1.01:
        dub.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-filter:a", f"atempo={tempo:.3f}", out])
    else:
        sf.write(out, y, sr)
    y2, _ = sf.read(out)
    return len(y2) / sr, dur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--thr", type=float, default=1.1)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--sim", type=float, default=0.70)
    args = ap.parse_args()
    name = os.path.splitext(os.path.basename(args.video))[0]
    wd = f"work/{name}"
    segs = json.load(open(f"{wd}/report_baseline.json", encoding="utf-8"))
    is_audio = os.path.splitext(args.video)[1].lower() in (".wav", ".mp3", ".m4a", ".flac")
    outp = f"{name}_hi_agent." + ("wav" if is_audio else "mp4")

    from sentence_transformers import SentenceTransformer, util
    emb = SentenceTransformer(r".\models\minilm", device="cpu")

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    path = r".\models\indic-parler-tts"
    model = ParlerTTSForConditionalGeneration.from_pretrained(path).to("cuda")
    tok = AutoTokenizer.from_pretrained(path)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate

    trace = []
    for i, s in enumerate(segs):
        s["overshoot"] = bool(s["tts_dur"] > s["avail"] * args.thr)
        if not s["overshoot"]:
            continue
        src_en = s["text"]
        best = {"text": s["text_hi"], "dur": s["tts_dur"]}
        log = {"seg": i, "english": src_en, "before_dur": s["tts_dur"], "avail": s["avail"], "attempts": []}
        cur = s["text_hi"]
        for t in range(args.tries):
            ratio = s["avail"] / max(best["dur"], 0.1)
            target = int(len(cur) * min(0.95, ratio * 0.95))
            cand = llm(f"Rewrite this Hindi dubbing line so it is shorter, about {target} characters (currently {len(cur)}). "
                       f"Keep the meaning and key terms. Output only the Hindi sentence.\n{cur}")
            back = llm(f"Translate this Hindi sentence into English. Output only the English.\n{cand}")
            sim = float(util.cos_sim(emb.encode(src_en), emb.encode(back)))
            ok_meaning = sim >= args.sim
            y = synth(model, tok, dtok, cand, seed=t)
            fitted, rawdur = fit(y, sr, s["avail"], f"{wd}/tts/{i:04d}.cand.wav")
            ok_time = fitted <= s["avail"] * args.thr
            log["attempts"].append({"try": t, "chars": len(cand), "sim": round(sim, 3), "fitted_dur": round(fitted, 2),
                                    "ok_meaning": ok_meaning, "ok_time": ok_time, "text_hi": cand, "back_en": back})
            if ok_meaning and fitted < best["dur"]:
                best = {"text": cand, "dur": fitted}
                os.replace(f"{wd}/tts/{i:04d}.cand.wav", f"{wd}/tts/{i:04d}.wav")
            if ok_meaning and ok_time:
                break
            cur = cand if ok_meaning else cur
        s["text_hi"] = best["text"]
        s["tts_dur"] = round(best["dur"], 2)
        s["overshoot"] = bool(best["dur"] > s["avail"] * args.thr)
        log["after_dur"] = s["tts_dur"]
        log["fixed"] = not s["overshoot"]
        trace.append(log)

    json.dump(segs, open(f"{wd}/report_agent.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(trace, open(f"{wd}/agent_trace.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del model
    gc.collect(); torch.cuda.empty_cache()
    bg = f"{wd}/sep/htdemucs/audio/no_vocals.wav"
    dub.stage_assemble(args.video, segs, bg, sr, wd, outp)
    n_flag = len(trace)
    n_fixed = sum(1 for t in trace if t["fixed"])
    n_left = sum(1 for s in segs if s["overshoot"])
    print(f"DONE -> {outp} | flagged: {n_flag} | fixed: {n_fixed} | overshoot left: {n_left}")

main()
