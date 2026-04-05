"""Auto-detect checkpoint type and load the right adapter."""

from typing import Tuple

import torch

from .base import ModelAdapter


def detect_checkpoint_type(path: str) -> str:
    """Detect model type from checkpoint contents."""
    checkpoint = torch.load(path, map_location='cpu')

    if checkpoint.get('model_type') in ('cql', 'iql'):
        return 'cql'

    if 'model_state_dict' in checkpoint and 'advantage_networks' not in checkpoint:
        return 'supervised'

    # Deep CFR: either top-level list keys or flattened per-network keys
    if 'strategy_networks' in checkpoint or 'advantage_networks' in checkpoint:
        return 'deep_cfr'
    if any(k.startswith('strategy_network_') or k.startswith('advantage_network_')
           for k in checkpoint):
        return 'deep_cfr'

    return 'unknown'


def load_model(path: str, exclude_fold: bool = False,
               **kwargs) -> Tuple[ModelAdapter, int]:
    """Load model from checkpoint, auto-detecting type.

    Returns (adapter, obs_size).
    """
    model_type = detect_checkpoint_type(path)

    if model_type == 'cql':
        from .cql_adapter import CQLAdapter
        adapter = CQLAdapter.from_checkpoint(path, exclude_fold=exclude_fold)
        return adapter, adapter.obs_size

    if model_type == 'deep_cfr':
        from .deep_cfr_adapter import DeepCFRAdapter
        adapter = DeepCFRAdapter.from_checkpoint(path, **kwargs)
        return adapter, adapter.obs_size

    if model_type == 'supervised':
        from .supervised_adapter import SupervisedAdapter
        adapter = SupervisedAdapter.from_checkpoint(path)
        return adapter, adapter.obs_size

    raise ValueError(f"Unknown checkpoint type at {path}")
