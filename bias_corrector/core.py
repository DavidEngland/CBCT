"""Core functions for simple ML-based bias correction.
This module includes a quantile-mapping implementation and a small wrapper
around a feed-forward regressor for residual correction.
"""
from typing import Tuple
import numpy as np
from sklearn.neural_network import MLPRegressor


def quantile_mapping_fit(obs: np.ndarray, sim: np.ndarray, q: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a quantile mapping between simulated and observed arrays.

    Returns (sim_quantiles, obs_quantiles) arrays of length q.
    """
    assert obs.shape == sim.shape
    probs = np.linspace(0, 1, q)
    sim_q = np.quantile(sim, probs)
    obs_q = np.quantile(obs, probs)
    return sim_q, obs_q


def quantile_mapping_apply(sim: np.ndarray, sim_q: np.ndarray, obs_q: np.ndarray) -> np.ndarray:
    """Apply quantile mapping to simulated values.

    Values outside the fitted range are linearly extrapolated.
    """
    # compute percentile for each sim value
    # use searchsorted on sim_q which is sorted
    idx = np.searchsorted(sim_q, sim, side="left")
    idx = np.clip(idx, 1, len(sim_q)-1)
    # linear interpolation between quantiles
    x0 = sim_q[idx-1]
    x1 = sim_q[idx]
    y0 = obs_q[idx-1]
    y1 = obs_q[idx]
    # avoid division by zero
    denom = (x1 - x0)
    denom_safe = np.where(denom == 0, 1e-8, denom)
    frac = (sim - x0) / denom_safe
    return y0 + frac * (y1 - y0)


def train_nn_correction(sim: np.ndarray, obs: np.ndarray, hidden_layer_sizes=(50,), random_state=0) -> MLPRegressor:
    """Train an MLP regressor to predict the residual (obs - sim) from sim.

    This is a simple baseline neural bias corrector. Inputs are 1D arrays of same length.
    Returns the trained sklearn MLPRegressor.
    """
    assert sim.shape == obs.shape
    X = sim.reshape(-1, 1)
    y = (obs - sim)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=random_state)
    model.fit(X, y)
    return model


def apply_nn_correction(sim: np.ndarray, model: MLPRegressor) -> np.ndarray:
    """Apply trained MLP to correct `sim` values by adding predicted residuals."""
    X = sim.reshape(-1, 1)
    pred = model.predict(X)
    return sim + pred
