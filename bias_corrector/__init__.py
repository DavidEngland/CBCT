"""CBCT: Bias Correction for Climate and Weather Models - package init
"""
from .core import quantile_mapping_fit, quantile_mapping_apply, train_nn_correction, apply_nn_correction

__all__ = [
    "quantile_mapping_fit",
    "quantile_mapping_apply",
    "train_nn_correction",
    "apply_nn_correction",
]
