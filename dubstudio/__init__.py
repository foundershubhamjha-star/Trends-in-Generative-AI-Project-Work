"""Dubbing Studio: an agentic English -> Indic dubbing pipeline.

Heavy dependencies (torch, transformers, faster-whisper) are imported lazily inside the
modules that need them, so importing this package is cheap and the pure-logic modules
(``segments``, ``config``) can be unit-tested without a GPU.
"""

__version__ = "2.1.0"
