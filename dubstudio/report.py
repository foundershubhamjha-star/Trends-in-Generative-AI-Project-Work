"""Presentation helpers: turn a pipeline result into table rows, metric cards and trace text.

Pure Python (no Gradio, no GPU) so the logic is unit-tested.
"""
import html

from .segments import classify, tempo_for

TABLE_HEADERS = ["#", "Start", "English", "Dubbed line (editable)", "Speed", "Status"]


def fmt_time(seconds):
    """``75.4`` -> ``'1:15'``."""
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def line_status(i, result):
    """Human-readable status of line ``i``: ok, fixed by agent, still long, or artifact."""
    thr = result["context"]["thr"]
    after = result["lines"][i]
    before = result["baseline_lines"][i]
    kind_before = classify(before["raw_dur"], before["avail"], thr)
    kind_after = classify(after["raw_dur"], after["avail"], thr)
    if kind_before == "artifact":
        return "timing artifact"
    if kind_before == "fixable" and kind_after == "ok":
        return "fixed by agent"
    if kind_after == "fixable":
        return "still long"
    return "ok"


def table_rows(result):
    """Rows for the editable line table (see :data:`TABLE_HEADERS`)."""
    rows = []
    for i, ln in enumerate(result["lines"]):
        rows.append([i + 1, fmt_time(ln['start']), ln["text"],
                     ln["text_tgt"], f"{tempo_for(ln['raw_dur'], ln['avail']):.2f}x",
                     line_status(i, result)])
    return rows


def edited_texts(table):
    """Extract the editable column from a table (list of rows or a pandas DataFrame)."""
    if hasattr(table, "values"):
        table = table.values.tolist()
    return [str(row[3]) for row in table]


def metric_cards(summary):
    """HTML for the headline metric cards."""
    def card(value, label):
        return (f'<div class="metric"><div class="metric-value">{html.escape(str(value))}</div>'
                f'<div class="metric-label">{html.escape(label)}</div></div>')
    cards = [
        card(summary["lines"], "lines dubbed"),
        card(f"{summary['fixed']} / {summary['fixable_before']}", "too-long lines fixed"),
        card(f"{summary['mean_tempo_before']:.2f}x → {summary['mean_tempo_after']:.2f}x",
             "mean speed-up"),
        card(summary["artifacts"], "timing artifacts skipped"),
        card(f"{summary['seconds']:.0f} s", "processing time"),
    ]
    return '<div class="metrics">' + "".join(cards) + "</div>"


def trace_markdown(trace):
    """Readable log of what the agent tried for every line it touched."""
    if not trace:
        return "_The agent had nothing to do: every line already fits its time slot._"
    out = []
    for t in trace:
        verdict = "fixed" if t.get("fixed") else "not fixed"
        out.append(f"**Line {t['line'] + 1}** — {verdict}  \n"
                   f"> {t['english']}  \n"
                   f"speech {t['before_dur']} s in a {t['avail']} s slot → {t['after_dur']} s")
        for s in t["steps"]:
            if s.get("error"):
                out.append(f"- `{s['strategy']}` failed: {s['error']}")
            elif s["strategy"] == "nbest":
                mark = "fits" if s["ok_time"] else "too long"
                out.append(f"- `n-best` candidate of {s['chars']} chars → {s['raw_dur']} s ({mark})")
            else:
                why = s.get("rejected")
                if why:
                    out.append(f"- `llm` try {s['try'] + 1}: rejected ({why})")
                else:
                    ok = "meaning kept" if s.get("ok_meaning") else "meaning check failed"
                    out.append(f"- `llm` try {s['try'] + 1}: {s['chars']} chars, "
                               f"similarity {s['sim']}, {ok}, {s['raw_dur']} s")
        out.append("")
    return "\n".join(out)
