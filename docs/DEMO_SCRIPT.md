# Demo video script (about 6 minutes, 4 test cases)

Record the screen with the UI open. Say the numbers from your own run; do not read the ones below as promises. Use headphones so the audio is clean, and show each result by playing a few seconds of original then dubbed.

**0:00 Intro (20 s).** "This is Dubbing Studio. It takes English speech and returns it in Hindi, keeps the background, and uses an agent to make sure each line fits its time slot. Everything runs offline on one 12 GB GPU."

**Case 1: a talk, audio only (about 2 min).** Upload your 6-minute talk. While it runs, name the stages shown in the progress bar. When done:
- *Listen* tab: play original, then the baseline, then the agent version around a line the agent fixed.
- Metric cards: say how many lines were too long and how many were fixed.
- *Agent trace* tab: open one fixed line and explain the steps ("n-best candidate, spoken, timed, accepted").
- Point out a line marked *timing artifact* and say why the agent leaves it.

**Case 2: a video (about 1 min).** Upload an mp4 with speech. Show that the dubbed result is a video with the new audio. Play 10 seconds.

**Case 3: edit a line (about 1 min).** In *Lines & edit*, change one Hindi line, press *Apply edits*, play the edited result. Say that only the changed line was re-spoken.

**Case 4: bad input handled gracefully (about 1 min).** Upload (a) a silent file, then (b) the Hindi dub from an earlier run, then (c) a file with a wrong extension (rename a text file to .mp3). Show the three clear messages. Then show the edge-case table in the README.

**Another language (30 s).** Choose Marathi or Tamil, run a short clip, and say plainly: "this language is marked experimental because I can only verify Hindi by ear."

**Close (20 s).** "Limits: English source only, one voice for all speakers, Hindi is the only language I validated. I also tested a memory layer inspired by A-MEM and Memp; it did not help, so I left it out and documented that."
