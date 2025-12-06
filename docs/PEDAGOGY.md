# Pedagogical Notes

This repository (`CBCT`) is designed as a teaching resource to introduce students
and researchers to practical bias correction techniques used in weather and climate modeling.

Contents:
- `bias_corrector/core.py`: minimal implementations of quantile mapping and a simple MLP-based residual corrector.
- `notebooks/01-intro.ipynb`: a reproducible walkthrough using synthetic data.
- `examples/generate_synthetic.py`: synthetic data generator for experiments.

Teaching suggestions:
- Start with `01-intro.ipynb` to understand the bias sources and how quantile mapping works.
- Ask students to compare quantile mapping vs. the NN residual corrector across seasons or percentiles.
- Extend the notebooks to use real climate data (NetCDF) and examine spatial patterns with `xarray`.

Further reading:
- Cannon, A.J., Best, A.C., et al. (2015) - Quantile mapping approaches
- Vrac, M., and Friederichs, P. (2015) - Bias correction of climate model outputs
