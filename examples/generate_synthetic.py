"""Generate synthetic observed and simulated time-series for demos."""
import numpy as np
import pandas as pd


def generate(n=1000, seed=0):
    rs = np.random.RandomState(seed)
    t = pd.date_range("2000-01-01", periods=n, freq="H")
    # true signal: diurnal + trend
    signal = 10 + 5 * np.sin(2 * np.pi * (np.arange(n) % 24) / 24.0) + 0.01 * np.arange(n)
    # obs = signal + noise
    obs = signal + rs.normal(scale=1.0, size=n)
    # sim has bias (additive + multiplicative) and different noise
    sim = 1.05 * signal + 2.0 + rs.normal(scale=1.5, size=n)
    return t, obs, sim


if __name__ == '__main__':
    t, obs, sim = generate(1000)
    import matplotlib.pyplot as plt
    plt.plot(t[:200], obs[:200], label='obs')
    plt.plot(t[:200], sim[:200], label='sim')
    plt.legend()
    plt.show()
