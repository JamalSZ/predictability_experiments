# Predictability experiments

Python code base for the paper *"Efficient Entropy Rate Estimation for
Quantifying Intrinsic Predictability of Univariate Numeric Time Series"*:
Lempel-Ziv / Kontoyiannis entropy-rate estimation under tolerance matching
(|x_a − x_b| ≤ ε), the upper bound Π^max on predictability, effective
alphabet sizes, synthetic Markov generators, forecasting benchmarks and the
five experiments proposed in the audit draft (Section E).

```
predictability/
  estimators/          Λ algorithms: bf, rnlj (+ rnlj_numpy), ifi, iesj; entropy rate
  predictability.py    Π^max (inverts eq:final, 'paper' variant; classical 'fano' variant)
  alphabet.py          effective alphabet size N (range / range_padded / recurrence / recurrence_permuted)
  synthetic.py         N_x D_y Markov chains, exact entropy rate, exact optimal predictability
  datasets.py          local real-dataset loader driven by configs/datasets.yaml (no downloads)
  benchmarks/runner.py statsforecast rolling one-step-ahead benchmarks, FOMM, ε-accuracy
  experiments/         exp1 … exp5 runners (CSV + PNG output)
  cli.py               command line entry point
configs/datasets.yaml  template – point it at your local CSV files
tests/                 pytest suite (algorithm agreement, formulas, pipeline)
scripts/               convenience shell scripts
```

## Installation

```bash
pip install -r requirements.txt      # numpy, scipy, pandas, matplotlib, pyyaml, statsforecast, numba, pytest
pip install -e .                     # optional, makes `predictability` importable anywhere
python -m pytest -q                  # 72 tests
```

## Real datasets (local only)

Nothing is downloaded. Edit `configs/datasets.yaml` and replace the
placeholder paths/columns with your local files (CSV/TXT/Parquet/NPY). Each
entry: `name`, `path`, `column` (name or 0-based index), optional
`season_length` (used by `SeasonalNaive` and the seasonal Auto* models),
`max_n` (hard cap 5000 for now), `slice: head|tail`, `sep`, `header`.

## Running the experiments

All commands write `*.csv` results, `*.png` plots and a `metadata.json` to
`--out`. Use `--help` on any sub-command for the full option list.

| Experiment | Command | Needs real data |
|---|---|---|
| 1 Upper-bound validity (synthetic, ε = 0) | `python -m predictability.cli exp1 --n 5000` | no |
| 2 Entropy-estimation accuracy vs exact rate | `python -m predictability.cli exp2 --n 5000` | no |
| 3 Runtime & peak memory (BF/RNLJ/IESJ/IFI) | `python -m predictability.cli exp3 --sizes 500,1000,2000,5000` <br> add `--config configs/datasets.yaml --datasets Temperature` for a real series | optional |
| 4 ε-sensitivity of Π^max on real data | `python -m predictability.cli exp4 --config configs/datasets.yaml` | yes |
| 5 Model-independence scatter + discretisation ablation | `python -m predictability.cli exp5 --config configs/datasets.yaml` (synthetic chains are added automatically; `--synthetic ""` to skip) | optional |
| Forecast benchmarks only (predictions CSV + ε-accuracy) | `python -m predictability.cli benchmark --config configs/datasets.yaml --eps 0.5,1,2` | yes |
| Entropy / N / Π^max table only | `python -m predictability.cli estimate --config configs/datasets.yaml --eps 0,0.5,1` | yes |

`scripts/run_all_synthetic.sh` runs everything that needs no real data;
`scripts/run_all_real.sh [config]` runs exp3 (real series), exp4 and exp5.

Common options: `--boundary lmp+|zero-fill` (default `lmp+`),
`--variant paper|fano` (Π^max formula, default `paper` = eq:final),
`--alphabet range|range_padded|recurrence|recurrence_permuted` (primary N
definition for Π^max; `range` and `recurrence` are always reported side by
side), `--algorithm` (Λ algorithm, default `ifi`), `--strict-eps`
(accuracy with `<` instead of `≤`).

### Output format of the benchmarks

`predictions_<dataset>.csv` is long format with columns
`dataset, model, t, true, predicted` (`t` = 0-based position in the series).
Accuracy for a tolerance ε is

```
accuracy(ε) = count(|true − predicted| ≤ ε) / number of predictions
```

(`predictability.benchmarks.eps_accuracy`; `strict=True` gives `<`).
Models come from [Nixtla statsforecast](https://github.com/Nixtla/statsforecast)
(`Naive`, `SeasonalNaive`, `WindowAverage`, `AutoARIMA`, `AutoETS`,
`AutoTheta`, `AutoCES`, …, selectable with `--models`), plus a first-order
Markov predictor `FOMM` for synthetic data. Evaluation protocol: 80/20 split,
rolling one-step-ahead forecasts over the test part via
`StatsForecast.cross_validation(h=1, step_size=1, refit=False)` — parameters
are fitted once on the training part and rolled forward (`--refit True`
re-fits at every step, `--refit k` every k steps). Models without a
`forward` method (e.g. `WindowAverage`) are always re-fit.
Other libraries (neuralforecast, darts, …) can be plugged in through
`MODEL_PROVIDERS` in `benchmarks/runner.py`; none are shipped.

## Library use

```python
from predictability.estimators import compute_lambda, estimate_entropy_rate
from predictability.alphabet import effective_alphabet_size
from predictability.predictability import pi_max

H = estimate_entropy_rate(x, eps=0.5, algorithm="ifi", boundary="lmp+")
N = effective_alphabet_size(x, 0.5, method="range")
print(pi_max(H, N, variant="paper"))
```

## Implementation notes and discrepancies found in the source documents

* **IFI lookup range (draft Section D).** The draft text and pseudocode query
  the index for keys in `[x_t − 2ε, x_t + 2ε]`, derived from an
  interval-overlap argument. This contradicts the matching definition
  `x_t ≈_ε x_t' ⇔ |x_t − x_t'| ≤ ε` used by the DP, the baseline and the
  entropy estimator; with a 2ε range IFI would return a *different* Λ than
  RNLJ/BF. The implementation uses `±ε` (and re-checks each candidate with
  `|x_t − k| ≤ ε` after a slightly widened binary search to make floating
  point rounding harmless). All algorithms are tested to produce identical Λ.
* **0-fill condition in Algorithm alg:baseline (main.tex).** The pseudocode
  tests `curr[t'] ≤ n`, which is always true. The text states λ_t = 0 if
  `t + max M[t,t'] > n`; that is what is implemented (`t + curr[t'] ≤ n`).
* **Π^max formula (eq:final).** Implemented exactly as written
  (`variant="paper"`): `H ≤ −Π(log₂Π − 1) − (1−Π)(log₂(1−Π) − log₂(N−2))`,
  inverted with `scipy.optimize.brentq` on the decreasing branch
  `[2/N, 1]`. Consequences worth knowing: `f(1) = 1` bit, so Π^max = 1 for any
  `H ≤ 1`; if `H ≥ f(2/N)` the result is clipped to `2/N` (random guessing)
  and flagged in a `*_status` column; `N ≤ 2` is degenerate (Π^max = 1). The
  classical Fano form (`H_b(Π) + (1−Π) log₂(N−1)`, branch `[1/N, 1]`) is
  available as `variant="fano"` and is used for the Song-et-al.-style
  discretisation ablation in Experiment 5.
* **Effective alphabet size.** `range`: `N = ⌈(x_max − x_min)/ε⌉` (main.tex
  writes it without the ceiling; the reviewer annotation asks for the
  ceiling). `range_padded`: `⌈(x_max + ε − (x_min − ε))/ε⌉` (KDD'24 form,
  audit G2). `recurrence`: eq:distN / eq:numericdist; `recurrence_permuted`:
  mean over random permutations with the 95 % half-width `1.96·s/√R`.
  Worked example check: the symbolic example (`bcbadbcd` → 4) is reproduced;
  in the numeric Example ex:numericalphasize the table lists `d_8 = 5`
  (total 25/8 = 3.125) but the formula gives `d_8 = 4` because
  `|5.0 − 4.1| = 0.9 ≤ 1`, i.e. 24/8 = 3.0 (and 3.625 instead of 3.75 for the
  third ordering). The code follows the formula; the example table in the
  paper should be corrected. In eq:numericdist the first case should read
  `t_k − max{…}` (distance), not `max{…}`, and the second case misses a
  closing parenthesis (audit G11).
* **Synthetic D_i family.** `p_w = 1/(10·i·N_s)` is undefined for `i = 0`;
  `i = 0` is implemented as the uniform chain (`p_w = 1/N_s`), the natural
  maximum-entropy end of the family.
* **ε-accuracy `≤` vs `<`.** The paper's Definition def:error counts a
  prediction as correct when `|x − x̂| ≤ ε`; the accuracy function defaults
  to `≤` (needed for ε = 0 exact matching on synthetic data). The strict
  version `count(|true − predicted| < ε)/n` requested in the discussion is
  available with `strict=True` / `--strict-eps`.
* **Experiment 1 at ε = 0.** Continuous forecasts never match a state value
  exactly, so benchmark predictions are snapped to the nearest state value
  (`--no-snap` disables this). FOMM and the exact Bayes-optimal predictability
  of the chain are always reported.
* **Experiment 3.** BF, RNLJ, IESJ and IFI are compared as pure-Python
  implementations (same language, same DP). The row-vectorised
  `rnlj_numpy` is available via `--algorithms`. Time (median of `--repeats`)
  and peak memory (`tracemalloc`, separate run) are measured separately.
  ε levels are pair-distance quantiles: `eps_min = 0`, `eps_25%`, … make
  ~25 %, … of all pairs match, `eps_max = x_max − x_min` makes all pairs match.
  BF is restricted to `n ≤ --bf-max-n` (default 1000). FIT (forest of
  interval trees) is not implemented (O(n³) space).
* **n ≤ 5000.** All loaders and generators cap the series length at 5000 for
  now (`datasets.MAX_N_HARD_LIMIT`).
