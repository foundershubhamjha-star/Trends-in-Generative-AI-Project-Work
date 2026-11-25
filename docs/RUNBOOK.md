# Runbook: from this folder to a submitted project

Run every command in PowerShell, one block at a time. Start each session with `cd C:\dubbing-studio` and `.\venv\Scripts\Activate.ps1` (or the Desktop launcher).

## 1. Install this package over the old one

Unzip `dubstudio_repo.zip` somewhere (for example Downloads), then:

```powershell
Copy-Item "$env:USERPROFILE\Downloads\dubstudio_repo\*" C:\dubbing-studio -Recurse -Force
```
```powershell
pip install pandas numpy
```
```powershell
python -m unittest discover -s tests -t .
```
Expected: `OK` at the end of the output.

## 2. First real run (find problems early)

```powershell
python -m dubstudio .\samples\clip2.wav --lang hi -v
```
Listen to `outputs\clip2_hi_baseline.wav` and `outputs\clip2_hi_agent.wav`.

## 3. Final evaluation on every sample

Put at least five different English files in `samples\` (talks, a lecture, an interview, a different accent; at least one 6-minute file and one video converted to wav is fine). Then:

```powershell
python -m dubstudio .\samples\*.wav --lang hi
```
```powershell
python tools\summarize_results.py --inject README.md
```
This fills the Results table in the README from `outputs\results.csv`.

## 4. Robustness suite

```powershell
python tools\edge_cases.py --base .\samples\clip2.wav --hindi .\outputs\clip2_hi_agent.wav --inject README.md
```
Takes a while (it runs about ten real jobs). It builds its inputs in `samples\edge\`. A `FAIL` row means a bug: send me the row and I will fix it before you record anything.

## 5. Web UI

```powershell
python app.py
```
Open http://127.0.0.1:7860. Try a file, then edit a line in *Lines & edit* and apply. Take two screenshots (the Listen tab and the Lines tab) and save them as `docs\ui_listen.png` and `docs\ui_lines.png`.

## 6. Publish to GitHub

Easiest: open your repo on github.com, delete the old files you no longer want, then use *Add file → Upload files* and drag in the contents of `C:\dubbing-studio` **except** `venv`, `models`, `work`, `samples`, `outputs`. (The `.gitignore` is only applied by git, not by the web uploader, so leave those folders out by hand.) Commit.

With git installed instead:

```powershell
git add -A
git commit -m "Dubbing Studio: packaged pipeline, agent, web UI, tests"
git push
```
Then: repo Settings → make it **Public**; Releases → create tag `v1.0`. Open the repo URL in a private window to confirm it loads, shows the README, and that `venv`/`models` are absent.

## 7. Videos

Scripts are in `DEMO_SCRIPT.md` and `WALKTHROUGH_SCRIPT.md`. Record with Windows Game Bar (Win+G) or OBS. Upload both to Google Drive, set *Share → Anyone with the link → Viewer*, and open each link in a **private window** to confirm it plays. Put both links near the top of the README.

## 8. Day before the review

- Restart the PC, open the launcher, run `python app.py`, dub one short file end to end.
- Start Ollama beforehand (Start menu) so the agent's LLM step is available.
- Close Chrome and Firefox before dubbing (they hold GPU memory).
- Read `REVIEW_CHEATSHEET.md` once, out loud.
