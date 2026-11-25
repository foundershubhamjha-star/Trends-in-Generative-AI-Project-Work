"""Exceptions raised by the pipeline.

Every exception derives from :class:`DubStudioError` and carries a message that is safe to
show directly to an end user (CLI or UI). Anything else is a bug.
"""


class DubStudioError(Exception):
    """Base class for all expected, user-presentable failures."""


class InputError(DubStudioError):
    """The input file is missing, unreadable, too short/long, or has no audio."""


class NoSpeechError(DubStudioError):
    """No speech was detected (silence, music only, or noise)."""


class UnsupportedLanguageError(DubStudioError):
    """The source audio is not in a supported language."""
