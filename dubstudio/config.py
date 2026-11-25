"""Central configuration: paths, thresholds, and the supported-language table.

Paths can be overridden with the environment variables ``DUBSTUDIO_MODELS``,
``DUBSTUDIO_WORK`` and ``DUBSTUDIO_OUT``.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from .errors import InputError

# --- model folder names (inside the models directory) ---
WHISPER_DIR = "faster-whisper-large-v3"
TRANSLATE_DIR = "indictrans2-en-indic"        # English -> Indic
TRANSLATE_REV_DIR = "indictrans2-indic-en"    # Indic -> English
TTS_DIR = "indic-parler-tts"
EMBED_DIR = "minilm"

# --- timing rules ---
MAX_TEMPO = 1.35          # largest speed-up applied to make a line fit its slot
DEFAULT_THR = 1.15        # a line may need up to this speed-up before the agent steps in
MIN_SLOT = 0.6            # seconds; slots shorter than this cannot hold a spoken line
MAX_RATIO = 4.0           # speech > 4x its slot is a timing artifact, not a length problem
MAX_INPUT_MINUTES = 30    # longer inputs are rejected with a clear message

# --- local LLM (Ollama) used by the agent's rewrite strategy ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("DUBSTUDIO_LLM", "qwen2.5:7b-instruct")

VOICE_TEMPLATE = ("{speaker}'s voice is monotone yet slightly fast in delivery, with a very "
                  "close recording that almost has no background noise.")


def models_dir() -> Path:
    return Path(os.environ.get("DUBSTUDIO_MODELS", "models"))


def work_dir() -> Path:
    return Path(os.environ.get("DUBSTUDIO_WORK", "work"))


def out_dir() -> Path:
    return Path(os.environ.get("DUBSTUDIO_OUT", "outputs"))


@dataclass(frozen=True)
class Language:
    """One target language.

    Attributes:
        code: short code used on the command line (e.g. ``hi``).
        name: English name, used inside LLM prompts.
        indictrans: IndicTrans2 language tag.
        speaker: Indic Parler-TTS speaker name used in the voice description.
        verified: True only if the output quality was checked by a fluent listener.
        llm_rewrite: whether the LLM rewrite strategy is trusted for this language.
    """
    code: str
    name: str
    indictrans: str
    speaker: str
    verified: bool = False
    llm_rewrite: bool = False

    @property
    def voice(self) -> str:
        return VOICE_TEMPLATE.format(speaker=self.speaker)


# English and Hindi are checked by ear. The other languages are experimental: the speaker
# names must be confirmed against the Indic Parler-TTS model card before relying on them.
LANGUAGES = {
    "en": Language("en", "English", "eng_Latn", "Mary", verified=True, llm_rewrite=True),
    "hi": Language("hi", "Hindi", "hin_Deva", "Divya", verified=True, llm_rewrite=True),
    "mr": Language("mr", "Marathi", "mar_Deva", "Sunita"),
    "bn": Language("bn", "Bengali", "ben_Beng", "Aditi"),
    "ta": Language("ta", "Tamil", "tam_Taml", "Jaya"),
    "te": Language("te", "Telugu", "tel_Telu", "Lalitha"),
}


def get_language(code: str) -> Language:
    """Return the :class:`Language` for ``code`` or raise :class:`InputError`."""
    try:
        return LANGUAGES[code]
    except KeyError:
        raise InputError(f"Unsupported target language '{code}'. "
                         f"Choose one of: {', '.join(sorted(LANGUAGES))}.") from None


@dataclass(frozen=True)
class Direction:
    """A validated translation direction."""
    source: Language
    target: Language
    model_dir: str          # translation model folder inside the models directory

    @property
    def key(self) -> str:
        return f"{self.source.code}>{self.target.code}"


def get_direction(source: str, target: str) -> Direction:
    """Validate ``source`` -> ``target`` and pick the translation model.

    Supported: English to any Indian language, and any Indian language to English. Raises
    :class:`InputError` for the same language on both sides or Indic-to-Indic.
    """
    src, tgt = get_language(source), get_language(target)
    if src.code == tgt.code:
        raise InputError("The source and target language are the same.")
    if src.code != "en" and tgt.code != "en":
        raise InputError("Only English to Indian-language and Indian-language to English "
                         "dubbing is supported. Choose English as the source or the target.")
    folder = TRANSLATE_DIR if src.code == "en" else TRANSLATE_REV_DIR
    return Direction(src, tgt, folder)


def directions():
    """All supported directions as ``(label, key)`` pairs for menus; Hindi and English first."""
    out = []
    for code, lang in LANGUAGES.items():
        if code == "en":
            continue
        tag = "" if lang.verified else " (experimental)"
        out.append((f"English \u2192 {lang.name}{tag}", f"en>{code}"))
    for code, lang in LANGUAGES.items():
        if code == "en":
            continue
        tag = "" if lang.verified else " (experimental)"
        out.append((f"{lang.name} \u2192 English{tag}", f"{code}>en"))
    return out
