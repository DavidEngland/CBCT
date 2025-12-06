import numpy as np
from bias_corrector.core import quantile_mapping_fit, quantile_mapping_apply, train_nn_correction, apply_nn_correction


def test_quantile_mapping_reduces_bias():
    rng = np.random.RandomState(0)
    n = 1000
    sim = rng.normal(loc=12.0, scale=2.0, size=n)
    obs = rng.normal(loc=10.0, scale=2.0, size=n)
    sim_q, obs_q = quantile_mapping_fit(obs, sim)
    corrected = quantile_mapping_apply(sim, sim_q, obs_q)
    # mean bias should be closer to zero after correction
    before = np.abs(sim.mean() - obs.mean())
    after = np.abs(corrected.mean() - obs.mean())
    assert after < before


def test_nn_correction_reduces_rmse():
    rng = np.random.RandomState(1)
    n = 1000
    signal = rng.normal(size=n)
    obs = signal + rng.normal(scale=0.5, size=n)
    sim = 1.2 * signal + 0.5 + rng.normal(scale=0.7, size=n)
    model = train_nn_correction(sim, obs)
    corrected = apply_nn_correction(sim, model)
    rmse_before = np.sqrt(np.mean((sim - obs)**2))
    rmse_after = np.sqrt(np.mean((corrected - obs)**2))
    assert rmse_after < rmse_before
