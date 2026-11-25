# Review cheat sheet

## The one-minute pitch
Dubbing Studio turns English speech into Hindi (and four experimental Indian languages), keeps the background, and uses an agent to fit each line into its time slot. The agent is a small verify-and-retry loop: it only accepts a fix after it has spoken and timed it.

## Why these choices (be ready to defend each)
- **Why an agent at all?** Hindi is longer than English, so lines overrun. A fixed speed-up makes speech unintelligible. The agent fixes the text instead of just speeding up.
- **Why cheapest-first (n-best before LLM)?** n-best candidates are direct translations, so meaning is safe and no LLM is needed. The LLM is the fallback because it can drift from the meaning. (Plan-and-Act, ReAct: act, observe, escalate.)
- **Why verify by speaking?** The only reliable measure of "fits" is the real synthesized duration, not character counts.
- **Why three checks on an LLM rewrite?** Each alone is weak: similarity misses dropped list items, the LLM judge is lenient on its own output, a length floor stops drastic cuts. Together they catch more. (Reflexion: feedback from the failed attempt goes into the next prompt.)
- **Why was memory removed?** I built a glossary and a rewrite memory (A-MEM, Memp style). On held-out data it gave no measurable benefit, the LLM glossary invented wrong terms, and bad accepted rewrites polluted the memory. I kept it as a documented negative result.
- **Why Demucs?** To keep music and effects instead of dubbing over silence.
- **Why local models?** Offline, free, reproducible on one 12 GB GPU; models load one at a time.
- **Why merge ASR fragments?** Tiny fragments translate badly and leave no time to speak. About 13% of lines in one talk were like that.
- **Why a 1.35x cap?** Beyond it speech becomes hard to understand. Remaining overruns are reported, not hidden.

## Likely questions
- *How do you evaluate quality?* Objective: fraction of too-long lines fixed, mean speed-up before and after. Subjective: listening checks in Hindi. No reference-based translation metric yet (no gold translations); say so.
- *Why does the agent sometimes fail?* Some lines are artifacts (slot under 0.6 s or speech over 4x the slot). Others have no shorter acceptable translation.
- *What about other languages?* The n-best step works through IndicTrans2; the voice names and quality are unverified because I cannot judge them. They are marked experimental in the UI.
- *Multiple speakers?* Not supported; one voice. Next step would be diarization.
- *Could you train something?* A LoRA-tuned small model to shorten Hindi lines is the natural next step (a 72 GB machine is available). Not done, and I say so.
- *Failure handling?* Show the edge-case table: silence, music, non-English, corrupt, empty, no-audio video, over 30 min, wrong language code.

## Things to admit up front
Only Hindi is validated by ear; the test set is small; the meaning check is a weak signal; one voice for all speakers.

## Before they open the app
Ollama running, venv active, Chrome closed, a short clip already processed (cached) in case time is tight.
