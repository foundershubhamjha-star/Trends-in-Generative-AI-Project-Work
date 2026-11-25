# Dubbing Studio

An agentic English → Indian-language dubbing pipeline that runs entirely offline on a single 12 GB GPU.

Course project for **Trends in Generative AI** (Prof. Mitesh Khapra, WSAI, IIT Madras).

Give it English speech (audio or video). It returns the same recording spoken in Hindi, with the original background sound kept and every line fitted to the time slot of the original. A small **agent** handles the hard part: Hindi is usually longer than English, so some translated lines would overrun their slot. The agent finds them, shortens them, checks the meaning survived, and re-speaks them.

## Features

- **Two interfaces:** a web UI (`python app.py`) and a batch command line (`python -m dubstudio`).
- **Audio and video in:** mp3, wav, m4a, flac, mp4, mkv, mov. Video is returned as video with the new audio track.
- **Background preserved:** Demucs separates voice from music and effects; only the voice is replaced.
- **Timing-aware agent:** detects lines that would need a large speed-up, then escalates from cheap to expensive fixes. Every candidate is spoken and timed before it is accepted.
- **Editable result:** in the UI, edit any Hindi line and re-speak just that line.
- **Honest comparison:** the un-fixed baseline is rendered alongside the agent version, so the effect of the agent can be heard.
- **Robust to bad input:** silence, music-only, non-English speech, corrupt files, over-long files and missing audio tracks all give a clear message instead of a crash (see [Robustness](#robustness-and-edge-cases)).
- **Both directions:** English to Hindi (checked by a fluent listener) and to Marathi, Bengali, Tamil and Telugu (experimental), plus Hindi to English (new, lightly tested).
- **Tested logic:** 54 unit tests cover merging, timing, classification and the agent (with fake models), the report helpers, and the re-render path.

## How it works

```
audio / video
  → ffmpeg: extract audio
  → Demucs htdemucs: split voice and background
  → faster-whisper large-v3: transcribe with timestamps, detect language
  → merge fragments into sentence-sized lines
  → IndicTrans2 (distilled 200M): English → target language
  → Indic Parler-TTS: speak each line, timed
  → agent: fix lines that are too long for their slot
  → ffmpeg: speed-fit (max 1.35x), mix over background, mux
```

### The agent

A line is **fixable** when its raw speech would need more than 1.15x speed-up to fit its slot. For each fixable line the agent escalates, cheapest first:

1. **n-best selection (no LLM).** IndicTrans2 proposes several translations of the same sentence. Shorter ones are spoken and timed. Every candidate is a direct translation, so the meaning is intact by construction. This works for all languages.
2. **LLM rewrite (Hindi only).** A local LLM (Qwen2.5 7B via Ollama) shortens the line. A rewrite is accepted only if it passes a **length floor** (it may not drop more than half the text), a **back-translation similarity** check, and an **LLM yes/no check** that key facts survived. A rejected attempt is fed back into the next prompt with the reason, and the temperature rises on each retry.

The agent never accepts a candidate that is not strictly shorter in spoken duration than the current best, so it cannot make a line worse. Lines whose slot is under 0.6 s, or whose speech is more than 4x the slot, are classed as **timing artifacts** (a transcript problem, not a length problem) and left alone.

Ideas come from the course papers: ReAct and Reflexion (act, verify, retry with feedback), Plan-and-Act (cheap strategies before expensive ones), and ToolRL / ReTool (tools called with verifiable outcomes). A-MEM and Memp inspired a memory layer that I built and measured; it did not help, and it is **not** part of the final pipeline (see [Development findings](#development-findings)).

## Models

| Stage | Model |
|---|---|
| Source separation | Demucs `htdemucs` |
| ASR | `Systran/faster-whisper-large-v3` |
| Translation | `ai4bharat/indictrans2-en-indic-dist-200M`, and `ai4bharat/indictrans2-indic-en-dist-200M` for Indian language to English (both gated: Hugging Face login needed) |
| TTS | `ai4bharat/indic-parler-tts` |
| Meaning check | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Agent LLM | `qwen2.5:7b-instruct` via Ollama |

Models load one at a time and are freed between stages, which is how the whole pipeline fits in 12 GB.

## Setup (Windows / PowerShell)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
winget install Gyan.FFmpeg
winget install Ollama.Ollama
ollama pull qwen2.5:7b-instruct
hf auth login
```

Download the models into `models/` (not committed):

```powershell
hf download Systran/faster-whisper-large-v3 --local-dir .\models\faster-whisper-large-v3
hf download ai4bharat/indictrans2-en-indic-dist-200M --local-dir .\models\indictrans2-en-indic
hf download ai4bharat/indic-parler-tts --local-dir .\models\indic-parler-tts
hf download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir .\models\minilm
```

`parler-tts` pins `transformers==4.46.1`, which is why `requirements.txt` holds `gradio<6` and `sentence-transformers<4`. Paths can be changed with the environment variables `DUBSTUDIO_MODELS`, `DUBSTUDIO_WORK`, `DUBSTUDIO_OUT`.

## Usage

**Web UI**

```powershell
python app.py
```

Opens at http://127.0.0.1:7860. Choose a file, pick a language, press *Dub it*. Tabs: *Listen* (original, dubbed, un-fixed baseline), *Lines & edit* (change any line and re-speak it), *Agent trace* (what the agent tried for each line).

**Command line**

```powershell
python -m dubstudio .\samples\clip2.wav --lang hi -v
python -m dubstudio .\samples\*.mp3 --lang hi          # batch; wildcards allowed
python -m dubstudio .\samples\talk.mp4 --lang hi       # video in, video out
python -m dubstudio .\samples\clip2.wav --no-agent     # baseline only
```

Options: `--lang {hi,mr,bn,ta,te}`, `--no-agent`, `--thr 1.15`, `--format {wav,mp3}`, `--skip-language-check`, `--out DIR`. Outputs go to `outputs/` (`<name>_<lang>_baseline.*` and `<name>_<lang>_agent.*`); a row per run is appended to `outputs/results.csv`; per-line detail is in `work/<name>/<lang>/report.json`. Each stage is cached, so a re-run only does new work.

**As a library**

```python
from dubstudio.pipeline import run
result = run("samples/clip2.wav", lang="hi")
print(result["summary"], result["outputs"])
```

<!-- REVERSE:START -->
## Hindi to English (reverse direction)

The same pipeline also dubs **Indian-language speech into English**. English output can be
checked by any listener, which makes this direction easy to verify.

- **Models:** speech recognition is the same Whisper model. Translation uses the matching
  `ai4bharat/indictrans2-indic-en-dist-200M` (gated, same Hugging Face login, but you must accept
  its access terms on its own page). English speech comes from Indic Parler-TTS, which supports
  English (speaker *Mary*).
- **Download once:**
  ```powershell
  hf download ai4bharat/indictrans2-indic-en-dist-200M --local-dir .\models\indictrans2-indic-en
  ```
- **Command line:** `python -m dubstudio .\samples\hindi_talk.wav --src hi --lang en`
- **Web UI:** choose *Hindi to English* in the Direction menu.
- **Agent:** English is usually shorter than Hindi, so far fewer lines overrun. When one does, the
  same n-best and LLM rewrite steps apply; the meaning check compares the English rewrite
  directly with the Hindi source using a multilingual embedding.
- **Supported directions:** English to any supported Indian language, and any supported Indian
  language to English. Indian language to Indian language is refused.
- **Status:** Hindi to English is new and lightly tested. Other source languages are experimental.
<!-- REVERSE:END -->

## Results

Metric: how many lines need more than 1.15x speed-up to fit their slot (**too long**), and how many the agent brings under that (**fixed**). Per-file numbers come straight from `outputs/results.csv`, summarised by `python tools/summarize_results.py --inject README.md`.

<!-- RESULTS:START -->
| File | Lines | Too long | Fixed | Still long | Timing artifacts | Mean speed-up | Time |
|---|---|---|---|---|---|---|---|
| clip2.wav | 12 | 2 | 2 | 0 | 0 | 1.06x → 1.01x | 43 s |
| clip.wav | 5 | 1 | 1 | 0 | 0 | 1.07x → 1.00x | 138 s |
| full.wav | 39 | 5 | 2 | 3 | 0 | 1.06x → 1.05x | 1754 s |
| speech2.wav | 28 | 3 | 2 | 1 | 0 | 1.03x → 1.02x | 1737 s |
| nehru.mp3 | 36 | 5 | 2 | 3 | 0 | 1.05x → 1.03x | 2037 s |
| jfk_english.mp4 | 14 | 1 | 0 | 1 | 0 | 1.03x → 1.03x | 417 s |
| clip2_hi_agent.wav | 4 | 0 | 0 | 0 | 0 | 1.00x → 1.00x | 108 s |
| hindi_short.mp4 | 3 | 0 | 0 | 0 | 0 | 1.00x → 1.00x | 143 s |
**Total:** the agent fixed 9 of 17 too-long lines across 8 files.
<!-- RESULTS:END -->

## Robustness and edge cases

`python tools/edge_cases.py --base samples\clip2.wav --hindi outputs\clip2_hi_agent.wav --inject README.md` builds awkward inputs from a clean clip and checks that each one ends in the expected outcome (success, or a specific user-facing error) and never in a crash.

<!-- EDGE:START -->
| Case | What it tests | Expected | Got | Result |
|---|---|---|---|---|
| `silence` | 10 s of digital silence | NoSpeechError | NoSpeechError | PASS |
| `music_only` | synthetic chord, no speech | NoSpeechError | NoSpeechError | PASS |
| `white_noise` | 10 s of noise | NoSpeechError | NoSpeechError | PASS |
| `short_3s` | 3-second excerpt of speech | ok | ok | PASS |
| `noisy_10dB` | speech with white noise at 10 dB SNR | ok | ok | PASS |
| `very_quiet` | speech at -26 dB | ok | ok | PASS |
| `fast_speech` | speech sped up 1.5x | ok | ok | PASS |
| `video_with_speech` | MP4 video + speech | ok | ok | PASS |
| `video_no_audio` | video with no audio track | InputError | InputError | PASS |
| `hindi_source` | Hindi speech given as the source | UnsupportedLanguageError | UnsupportedLanguageError | PASS |
| `corrupt_file` | random bytes named .mp3 | InputError | InputError | PASS |
| `empty_file` | zero-byte file | InputError | InputError | PASS |
| `too_long_31min` | 31 minutes (limit is 30) | InputError | InputError | PASS |
| `missing_file` | path that does not exist | InputError | InputError | PASS |
| `bad_language` | unsupported target language code 'xx' | InputError | InputError | PASS |

15 / 15 cases behaved as expected.
<!-- EDGE:END -->

## Development findings

These come from the first prototype (`legacy/`: `dub.py`, `agent2.py`, `agent3.py`) on a 6-minute and a 2.5-minute talk, plus a held-out talk. They are small samples and are reported as observations, not proof.

| Experiment | Outcome |
|---|---|
| 3B vs 7B rewriting LLM | 7B clearly better. Qwen3-4B's reasoning mode leaked into the output and was unusable. |
| Stricter agent (length floor, LLM judge, retry feedback) | On the held-out talk it fixed 3 of 3 fixable lines (earlier version: 3 of 4). |
| Memory of earlier rewrites (A-MEM / Memp style) | **No measurable benefit.** A 1-2 line difference is within noise. Polluted memory (accepted rewrites that dropped content) made one run worse. Removed from the final pipeline. |
| LLM-generated glossary | **Negative result.** The LLM mistranslated terms and invented names, even after validation. Removed. |
| LLM judging its own rewrite | Lenient. The judge is combined with a length floor and a back-translation check instead of being trusted alone. |
| Unfixable lines | About 13% of lines in the 6-minute talk were tiny ASR fragments with no usable slot. Fixed upstream by merging fragments, and the rest are classed as timing artifacts. |

## Known limitations

- Sources: English, and Hindi to English (new, lightly tested). Other source languages are refused with a clear message.
- One voice for all speakers; there is no speaker separation yet.
- Hindi is the only target language checked by ear. The LLM rewrite is disabled for the others; the n-best step still runs.
- The meaning check (back-translation plus embedding similarity plus an LLM yes/no) is a weak signal for Hindi.
- Speed-fitting is capped at 1.35x to keep speech intelligible, so a few lines may still run slightly long.
- Tested on Windows with an RTX A2000 (12 GB). Needs an NVIDIA GPU.

## Repository layout

```
app.py                     start the web UI
dubstudio/
  config.py                paths, thresholds, language table
  pipeline.py              orchestration: run() and redub()
  asr.py  translate.py  tts.py  audio.py   one module per stage
  agent.py                 the shortening agent
  segments.py              pure logic: merging, timing, classification
  report.py  ui.py         table/metric helpers and the Gradio interface
  cli.py  llm.py  gpu.py  errors.py
tests/                     54 unit tests (python -m unittest discover -s tests -t .)
tools/                     edge_cases.py, summarize_results.py, ui_preview.py
docs/                      RUNBOOK.md, DEMO_SCRIPT.md, WALKTHROUGH_SCRIPT.md, REVIEW_CHEATSHEET.md
legacy/                    first prototype scripts, kept for the development findings
```

## Acknowledgements

AI4Bharat (IndicTrans2, Indic Parler-TTS), OpenAI Whisper and SYSTRAN faster-whisper, Meta Demucs, Hugging Face, Ollama and Qwen. Course papers: ReAct, Reflexion, Plan-and-Act, A-MEM, Memp, MEM1, ACON, ToolRL, ReTool and others from the Trends in Generative AI reading list.

## License

MIT. See `LICENSE`.
