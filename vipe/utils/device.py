import torch


_runtime_device = torch.device("cuda")


def configure_device(device: str | torch.device) -> torch.device:
    """Set the process-wide device selected by the top-level ViPE config."""
    global _runtime_device
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported ViPE device: {selected}; expected 'cpu' or 'cuda'")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ViPE is configured for CUDA, but torch.cuda.is_available() is false")
    if selected.type == "cuda" and selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    _runtime_device = selected
    return selected


def get_device() -> torch.device:
    return _runtime_device
