"""Model adapters: unified interface for CQL, Deep CFR, and Supervised models."""

from .base import ModelAdapter
from .loader import load_model

__all__ = ['ModelAdapter', 'load_model']
