# Trends-in-Generative-AI-Project-Work
Dubbing Studio Orchestrator
# Dubbing Studio: an agentic English → Hindi dubbing pipeline

Course project for **Trends in Generative AI** (Prof. Mitesh Khapra, WSAI, IIT Madras).

**Status: work in progress.** The baseline pipeline and a first agent layer work end to end. Evaluation, multi-speaker support, a UI and model fine-tuning are in progress (see the roadmap). Results below are early and based on small test sets.

## What it does

Takes English speech (audio or video) and produces Hindi dubbed audio with the original background sound preserved. Each Hindi line is fitted to the time slot of the original line. When a translated line is too long to fit naturally, an LLM-based agent shortens it, checks that the meaning survived, and re-synthesizes the speech.

```
audio/video
  → extract audio (ffmpeg)
  → separate vocals / background (Demucs htdemucs)
  → speech recognition with timestamps (faster-whisper large-v3)
  → translation EN→HI (IndicTrans2 distilled 200M)
  → speech synthesis (Indic Parler-TTS) + time fitting (ffmpeg atempo, capped at 1.35x)
  → remix with background, mux into video/audio output
```

## Agent layer (`agent2.py`)

Runs on top of the baseline's saved outputs. For each line whose Hindi speech would need more than a set speed-up to fit its slot:

1. **Shorten:** an LLM (Ollama) rewrites the Hindi line to a target length.
2. **Verify meaning:** the rewrite is translated back to English and compared with the source using a multilingual sentence-embedding model (cosine similarity).
3. **Verify timing:** the rewrite is synthesized and its raw duration is compared with the slot.
4. **Retry** up to N times; keep the best verified candidate.

Optional memory (`--mem 1`), inspired by A-MEM and Memp:
- **Glossary:** consistent Hindi renderings for key terms.
- **Rewrite memory:** earlier successful shortenings are retrieved as few-shot examples for similar lines.

Design ideas come from the course papers: ReAct and Reflexion (act, check, retry), Plan-and-Act, A-MEM / Memp / MEM1 (memory), ACON (context compression), ToolRL / ReTool (tool-use training, planned).

## Models

| Stage | Model |
|---|---|
| Source separation | Demucs `htdemucs` |
| ASR | `Systran/faster-whisper-large-v3` |
| Translation | `ai4bharat/indictrans2-en-indic-dist-200M` (gated, needs Hugging Face login) |
| TTS | `ai4bharat/indic-parler-tts` |
| Meaning check | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Agent LLM | `qwen2.5:7b-instruct` via Ollama (see notes below) |

Developed on a single 12 GB GPU (RTX A2000). Models are loaded one at a time and freed between stages.

## Setup (Windows / PowerShell)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper demucs transformers sentencepiece sacremoses soundfile librosa huggingface_hub
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
pip install git+https://github.com/huggingface/parler-tts.git
pip install "gradio<6" "sentence-transformers<4"
winget install Gyan.FFmpeg
winget install Ollama.Ollama
ollama pull qwen2.5:7b-instruct
hf auth login
```

Download models into `models/` (not committed to the repo):

```powershell
hf download Systran/faster-whisper-large-v3 --local-dir .\models\faster-whisper-large-v3
hf download ai4bharat/indictrans2-en-indic-dist-200M --local-dir .\models\indictrans2-en-indic
hf download ai4bharat/indic-parler-tts --local-dir .\models\indic-parler-tts
hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir .\models\minilm
```

Version notes: `parler-tts` pins `transformers==4.46.1`, so `gradio` and `sentence-transformers` must be the older compatible releases shown above.

## Usage

```powershell
# 1. Baseline dub (audio or video input)
python dub.py .\samples\clip.wav
cp work\clip\report.json work\clip\report_baseline.json

# 2. Agent without / with memory (threshold = allowed speed-up ratio)
python agent2.py .\samples\clip.wav --mem 0 --thr 1.15
python agent2.py .\samples\clip.wav --mem 1 --thr 1.15
```

Outputs: `<name>_hi.wav` (baseline), `<name>_hi_agent.wav`, `<name>_hi_agentmem.wav`, with per-line logs in `work/<name>/` (`report_baseline.json`, `agent*/trace.json`, `agent*/summary.json`) and the shared memory in `work/memory.json`.

## Early results

Metric: number of lines whose Hindi speech needs more than 1.15x speed-up to fit the original slot, and how many the agent fixes. Segments under 0.6 s are skipped.

| Clip | Usable segments | Need > 1.15x (baseline) | Agent fixed (3B LLM) | Agent fixed (7B LLM) | Agent + memory fixed (7B) |
|---|---|---|---|---|---|
| Speech clip, 6 min | 47 | 12 | 3 | 5 of 12 | 6-7 of 13 |
| Speech clip, 2.5 min | 12 | 2 | n/a | 2 of 2 | 1 of 2 |
| Held-out speech | 37 segments | pending | pending | pending | pending |

**Honest reading of these numbers:**
- Moving the rewriter from a 3B to a 7B model clearly helped. Using Qwen3-4B was a dead end because its reasoning mode leaks into the output.
- Memory showed no significant benefit so far (a 1-2 line difference is within noise).
- Inspecting the saved memory showed problems: some glossary entries have wrong Hindi (the LLM mistranslates terms), and a few accepted rewrites drop meaning even though they pass the embedding-similarity check (for example, long lists cut down to fragments). Because bad rewrites can be saved as examples, memory can hurt rather than help.
- Sample sizes are small; none of these comparisons is statistically meaningful yet.

## Known limitations

- Single Hindi voice for all speakers (no diarization yet).
- Meaning check uses back-translation plus embedding similarity, which is a weak signal for Hindi.
- LLM glossary entries are unreliable for non-name terms.
- Hindi is usually longer than English, so some lines need compression regardless.
- Evaluation so far is partly manual listening; no reference-based metrics yet.

## Roadmap

- [ ] Held-out test: fill memory on one talk, evaluate on another, with and without memory
- [ ] Stricter meaning gate (LLM judge plus length floor); only quality-gated rewrites go into memory
- [ ] Validate glossary entries (transliterate names, drop unreliable term renderings)
- [ ] IndicTrans2 multi-candidate selection as a no-LLM baseline for length fitting
- [ ] Evaluation script: speed-up ratio, ASR WER, chrF++ on a manually corrected sample, human check
- [ ] Multi-speaker support (diarization, per-speaker voice descriptions)
- [ ] Fine-tune a small LLM to shorten Hindi lines (LoRA/QLoRA, larger GPU), compare with the prompted baseline
- [ ] Gradio UI: upload, edit a line, regenerate, download
- [ ] Final report and demo video

## Repository layout

```
dub.py            baseline pipeline (stages cached under work/<name>/)
agent2.py         agent layer (shorten, verify, retry, optional memory)
README.md
.gitignore        excludes models/, work/, samples/, venv/, audio outputs
```

## Acknowledgements

AI4Bharat (IndicTrans2, Indic Parler-TTS), OpenAI Whisper and SYSTRAN faster-whisper, Meta Demucs, Hugging Face, Ollama and Qwen. Course papers: ReAct, Reflexion, Plan-and-Act, A-MEM, Memp, MEM1, ACON, ToolRL, ReTool and others from the Trends in Generative AI reading list.
