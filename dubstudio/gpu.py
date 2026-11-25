"""Small GPU-memory helper."""
import gc


def free() -> None:
    """Release cached GPU memory (no-op if torch or CUDA is unavailable)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def prepare_cuda_dlls() -> None:
    """Make CUDA DLLs bundled with PyTorch visible to CTranslate2 (faster-whisper) on Windows.

    faster-whisper needs ``cublas64_12.dll``. PyTorch ships it in its own ``lib`` folder, but
    Windows only finds it if that folder is registered or torch was imported first.
    """
    import os
    try:
        import torch  # noqa: F401  (importing torch loads its CUDA libraries)
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if hasattr(os, "add_dll_directory") and os.path.isdir(lib):
            os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
