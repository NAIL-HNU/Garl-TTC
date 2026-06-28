from __future__ import annotations

from pathlib import Path

import torch


def pick_device(device: str | None = None) -> torch.device:
    if device and device != 'auto':
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(model: torch.nn.Module, checkpoint: str | Path, device: torch.device, strict: bool = True) -> None:
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)
