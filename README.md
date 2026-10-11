# Generic k-SAT Numerical Optimization

Numerical implementation of the optimization procedure for the generalized biased-PPSZ analysis for Unique k-SAT.

The program supports `k = 3, 4, 5, 6, 7` and implements:

- k-dependent PPSZ numerical quantities;
- prefix-system coefficients and strengthened forcing caps;
- With-r-CVXPY feasibility optimization on an active threshold set;
- No-r-CVXPY optimization for fixed `(w, rho, t)`;
- dense threshold scanning and Golden Search refinement;
- active-set numerical optimization.

## Requirements

- Python 3.10+
- CVXPY
- NumPy
- SciPy
- CLARABEL
- SCS

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/generic-k-sat-numerical.git
cd generic-k-sat-numerical
python -m venv .venv
```

Activate the virtual environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python generic_k_sat.py
```

The script evaluates `k = 3, 4, 5, 6, 7` sequentially and prints:

- optimized weight vector `w`;
- bias parameters `rho`;
- final exponent;
- exponential base;
- minimizing threshold `t`;
- selected `gamma`.

## Main numerical parameters

The main numerical parameters are defined near the top of `generic_k_sat.py`:

- `T_STEP`
- `ACTIVE_STEP`
- `BISECTION_ITERS`
- `DUPLICATE_TOL`
- `GOLDEN_TOL`
- `GOLDEN_ITERS`
- `MAX_LOCAL_REFINEMENTS`

## Solver behavior

CLARABEL is the primary CVXPY solver. SCS is used as a fallback solver.

The With-r stage additionally performs explicit numerical feasibility checks to reduce the risk of misclassification near the feasibility boundary.

## Reproducibility

The default script runs `k = 3, 4, 5, 6, 7` in sequence. The exponent obtained for the current `k` is rounded upward to nine decimal places before being used as `R_(k-1)` for the next value of `k`.

## Citation

If you use this implementation in academic work, please cite the associated paper. Replace this section with the final paper citation or BibTeX entry when available.

## License

MIT License. See `LICENSE`.
