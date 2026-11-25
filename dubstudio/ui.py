"""Gradio web interface. Start it with ``python app.py``.

The pipeline functions are injected (``runner`` / ``redubber``) so the interface can be
exercised without a GPU.
"""
import html
import logging

import gradio as gr

from . import config
from .errors import DubStudioError
from .report import TABLE_HEADERS, edited_texts, metric_cards, table_rows, trace_markdown

log = logging.getLogger(__name__)

STAGE_LABELS = {"audio": "Extracting audio", "separate": "Separating voice from background",
                "transcribe": "Transcribing", "translate": "Translating",
                "synthesize": "Speaking", "agent": "Agent shortening lines",
                "render": "Mixing", "done": "Done"}

CSS = """
:root { --ink:#1f1b16; --muted:#7a6f61; --paper:#f7f2e9; --card:#fffdf8; --line:#e6dccb;
        --accent:#b4451a; --accent-soft:#f6e4da; --good:#2f7d4f; }
.gradio-container { max-width: 1240px !important; background: var(--paper) !important; }
footer { display: none !important; }
#hero { padding: 28px 4px 8px 4px; }
#hero h1 { font-family: Georgia, 'Times New Roman', serif; font-size: 44px; line-height: 1.05;
           letter-spacing: -0.02em; margin: 0; color: var(--ink); }
#hero h1 em { color: var(--accent); font-style: italic; }
#hero p { margin: 10px 0 14px 0; color: var(--muted); font-size: 16px; max-width: 640px; }
.chips span { display:inline-block; margin: 0 6px 6px 0; padding: 3px 11px; border-radius: 99px;
              background: var(--accent-soft); color: var(--accent); font-size: 12.5px;
              font-weight: 600; letter-spacing: .02em; }
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
         padding: 18px !important; }
.panel-title { font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
               color: var(--muted); margin: 0 0 6px 0; }
#dub-btn { background: var(--accent) !important; border: none !important; color: #fff !important;
           font-weight: 650; font-size: 16px; padding: 12px 0; border-radius: 10px; }
#dub-btn:hover { filter: brightness(1.08); }
.metrics { display:grid; grid-template-columns: repeat(5, 1fr); gap: 10px;
           margin: 4px 0 14px 0; }
.metric { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
          padding: 12px 14px; }
.metric-value { font-family: Georgia, serif; font-size: 21px; white-space: nowrap; color: var(--ink); }
.metric-label { font-size: 12px; color: var(--muted); margin-top: 2px; }
.empty { text-align:center; color: var(--muted); padding: 60px 20px; border: 1.5px dashed var(--line);
         border-radius: 14px; background: transparent; }
.empty b { color: var(--ink); font-family: Georgia, serif; font-size: 20px; font-weight: 400; }
.note { border-left: 3px solid var(--accent); background: var(--accent-soft); color: #6b2a10;
        padding: 8px 12px; border-radius: 0 8px 8px 0; font-size: 13.5px; margin: 6px 0; }
.status-ok { color: var(--good); font-weight: 600; }
.tabs button.selected { color: var(--accent) !important; border-color: var(--accent) !important; }
"""

HERO = """
<div id="hero">
  <h1>Dubbing <em>Studio</em></h1>
  <p>Drop in speech in English or an Indian language. Get it back in the other, in a natural
  voice, with the background kept and an agent making sure every line fits its time slot.</p>
  <div class="chips"><span>Whisper large-v3</span><span>Demucs</span><span>IndicTrans2</span>
  <span>Indic Parler-TTS</span><span>Local LLM agent</span><span>100% offline</span></div>
</div>
"""

EMPTY = ('<div class="empty"><b>Nothing dubbed yet</b><br>Choose a file on the left and press '
         '<i>Dub it</i>. A six-minute talk takes a few minutes on a 12 GB GPU.</div>')


def _pair(path, has_video, label):
    """Updates for an (audio player, video player) pair; exactly one is shown."""
    if not path:
        return gr.update(value=None, visible=False), gr.update(value=None, visible=False)
    if has_video:
        return gr.update(value=None, visible=False), gr.update(value=path, visible=True, label=label)
    return gr.update(value=path, visible=True, label=label), gr.update(value=None, visible=False)


def build_app(runner, redubber):
    """Create the Blocks app. ``runner`` is ``pipeline.run``, ``redubber`` is ``pipeline.redub``."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange, neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
        font=("Segoe UI", "system-ui", "sans-serif"),
    ).set(body_background_fill="#f7f2e9", block_background_fill="#fffdf8",
          block_border_color="#e6dccb", border_color_primary="#e6dccb",
          button_primary_background_fill="#b4451a", button_primary_text_color="white")

    force_light = "() => { document.body.classList.remove('dark'); }"
    with gr.Blocks(theme=theme, css=CSS, js=force_light, title="Dubbing Studio") as demo:
        state = gr.State(None)
        gr.HTML(HERO)
        with gr.Row(equal_height=False):
            # ------------------------------------------------------------ left: controls
            with gr.Column(scale=4, elem_classes="panel"):
                gr.HTML('<div class="panel-title">1 · Source</div>')
                file_in = gr.File(label="Audio or video with clear speech",
                                  file_types=["audio", "video"], type="filepath")
                gr.HTML('<div class="panel-title" style="margin-top:12px">2 · Options</div>')
                lang = gr.Dropdown(config.directions(), value="en>hi", label="Direction",
                                   interactive=True)
                lang_note = gr.HTML("")
                use_agent = gr.Checkbox(True, label="Shortening agent",
                                        info="Fixes lines that are too long for their time slot "
                                             "(also renders the un-fixed version to compare).")
                thr = gr.Slider(1.0, 1.5, value=config.DEFAULT_THR, step=0.05,
                                label="Agent trigger (speed-up ratio)",
                                info="Lower = stricter timing, more agent work.")
                dub_btn = gr.Button("Dub it", elem_id="dub-btn", variant="primary")
                status = gr.HTML("")
            # ----------------------------------------------------------- right: results
            with gr.Column(scale=8):
                placeholder = gr.HTML(EMPTY)
                metrics = gr.HTML("", visible=False)
                with gr.Tabs(visible=False) as tabs:
                    with gr.Tab("Listen"):
                        with gr.Row():
                            with gr.Column(elem_classes="panel"):
                                gr.HTML('<div class="panel-title">Original</div>')
                                orig_a = gr.Audio(label="Original", visible=False)
                                orig_v = gr.Video(label="Original", visible=False)
                            with gr.Column(elem_classes="panel"):
                                gr.HTML('<div class="panel-title">Dubbed</div>')
                                final_a = gr.Audio(label="Dubbed", visible=False)
                                final_v = gr.Video(label="Dubbed", visible=False)
                        with gr.Column(elem_classes="panel"):
                            gr.HTML('<div class="panel-title">Without the agent (for comparison)</div>')
                            base_a = gr.Audio(label="Baseline", visible=False)
                            base_v = gr.Video(label="Baseline", visible=False)
                    with gr.Tab("Lines & edit"):
                        gr.Markdown("Edit any dubbed line, then press **Apply edits**. Only "
                                    "changed lines are re-spoken.")
                        table = gr.Dataframe(headers=TABLE_HEADERS,
                                             datatype=["number", "str", "str", "str", "str", "str"],
                                             interactive=True, wrap=True,
                                             static_columns=[0, 1, 2, 4, 5],
                                             column_widths=["6%", "12%", "28%", "33%", "9%", "12%"],
                                             max_height=460)
                        apply_btn = gr.Button("Apply edits and re-render")
                        edit_msg = gr.HTML("")
                        edit_a = gr.Audio(label="Edited dub", visible=False)
                        edit_v = gr.Video(label="Edited dub", visible=False)
                    with gr.Tab("Agent trace"):
                        trace_md = gr.Markdown()
                    with gr.Tab("About"):
                        gr.Markdown(ABOUT)

        # ------------------------------------------------------------------ handlers
        def on_lang(key):
            d = config.get_direction(*key.split(">"))
            names = [x.name for x in (d.source, d.target) if not x.verified]
            if not names:
                return ""
            return ('<div class="note">%s has not been checked by a fluent listener yet. '
                    'The LLM rewrite step is disabled for it.</div>' % html.escape(" and ".join(names)))

        lang.change(on_lang, lang, lang_note)

        def on_dub(path, key, agent_on, threshold, progress=gr.Progress()):
            if not path:
                raise gr.Error("Please choose an audio or video file first.")
            cb = lambda stage, frac, msg="": progress(frac, desc=f"{STAGE_LABELS.get(stage, stage)} {msg}")
            try:
                src_code, tgt_code = key.split(">")
                res = runner(path, lang=tgt_code, source=src_code, use_agent=agent_on,
                             thr=threshold, progress=cb)
            except DubStudioError as exc:
                raise gr.Error(str(exc))
            except Exception as exc:  # last resort: never show a stack trace to the user
                log.exception("unexpected failure")
                raise gr.Error(f"Something went wrong: {exc}. Details are in the terminal.")
            ctx = res["context"]
            hv = ctx["has_video"]
            out = res["outputs"]
            final = out.get("agent") or out.get("dubbed")
            oa, ov = _pair(path, hv, "Original")
            ba, bv = _pair(out.get("baseline"), hv, "Baseline (no agent)")
            fa, fv = _pair(final, hv, "Dubbed")
            notes = "".join(f'<div class="note">{html.escape(w)}</div>' for w in res["warnings"])
            done = '<div class="status-ok">Done. Files saved in the outputs folder.</div>'
            return (state_value(res), gr.update(visible=False), gr.update(value=metric_cards(res["summary"]), visible=True),
                    gr.update(visible=True), oa, ov, ba, bv, fa, fv, table_rows(res),
                    trace_markdown(res["trace"]), done + notes,
                    gr.update(value=None, visible=False), gr.update(value=None, visible=False), "")

        dub_btn.click(on_dub, [file_in, lang, use_agent, thr],
                      [state, placeholder, metrics, tabs, orig_a, orig_v, base_a, base_v,
                       final_a, final_v, table, trace_md, status, edit_a, edit_v, edit_msg])

        def on_apply(df, res, progress=gr.Progress()):
            if not res:
                raise gr.Error("Dub a file first.")
            cb = lambda stage, frac, msg="": progress(frac, desc=f"{STAGE_LABELS.get(stage, stage)} {msg}")
            try:
                new = redubber(res, edited_texts(df), progress=cb)
            except DubStudioError as exc:
                raise gr.Error(str(exc))
            except Exception as exc:
                log.exception("re-render failed")
                raise gr.Error(f"Something went wrong: {exc}. Details are in the terminal.")
            res = dict(res, lines=new["lines"])
            ea, ev = _pair(new["outputs"]["edited"], res["context"]["has_video"], "Edited dub")
            n = len(new["changed"])
            msg = f'<div class="status-ok">{n} line{"s" if n != 1 else ""} re-spoken.</div>'
            return res, ea, ev, msg

        apply_btn.click(on_apply, [table, state], [state, edit_a, edit_v, edit_msg])

    demo.queue(default_concurrency_limit=1)   # one GPU job at a time
    return demo


def state_value(res):
    """What is kept in the session between clicks (everything is JSON-friendly)."""
    return res


ABOUT = """
**How it works.** Demucs splits the voice from the background. Whisper transcribes, IndicTrans2
translates, and Indic Parler-TTS speaks each line. Hindi is usually longer than English, so a
line can overrun its slot. The agent finds those lines, asks the translator for shorter
alternatives, and only if needed lets a local LLM rewrite the line. Each candidate is spoken and
timed before it is accepted, and a rewrite must also pass a meaning check.

**Limits.** English to Indian languages, or Indian languages to English; nothing between two
Indian languages. English and Hindi are the only languages checked by a fluent listener. One
speaker voice is used for everything. Files over 30 minutes are rejected.
"""


def launch(runner=None, redubber=None, port=7860, inbrowser=True):
    """Start the web UI on localhost."""
    if runner is None or redubber is None:
        from .pipeline import redub, run
        runner, redubber = runner or run, redubber or redub
    app = build_app(runner, redubber)
    app.launch(server_name="127.0.0.1", server_port=port, inbrowser=inbrowser,
               show_api=False, max_file_size="600mb")
