# CBCT — Climate & Weather Bias Correction Toolkit

This repository provides educational, reproducible tools for integrating
machine-learning-based bias correction into weather and climate workflows.

What you'll find:
- `bias_corrector/` — minimal Python package with baseline bias-correction methods.
- `notebooks/` — reproducible Jupyter notebooks (start with `01-intro.ipynb`).
- `examples/` — small scripts to generate synthetic data for demos.
- `docs/PEDAGOGY.md` — teaching notes and suggested exercises.
- `tests/` — basic unit tests demonstrating correctness improvements.

Goals
- Provide clear, minimal reference implementations (quantile mapping and an MLP residual corrector).
- Offer reproducible notebooks and visualizations for teaching and research.
- Make it easy to extend to real NetCDF datasets and spatial bias correction.

Getting started
1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the example notebook with Jupyter:

```bash
jupyter notebook notebooks/01-intro.ipynb
```

Contributing
- Open issues or PRs for new correction methods, spatial extensions, or dataset recipes.

License
- This project is open-source. Add a license file if you plan to publish publicly.
Climate Bias Correction Tool
