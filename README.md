# Residual Trading Strategy from an Interacting Agent-Based Model

A research project that builds a **two-state interacting agent-based model (ABM)** of
financial markets — in the spirit of Kaizoji (2000, 2006) — and turns its
one-step-ahead forecast errors into a **residual-based trading signal** on the
crypto cross-section, priced through an **moving average-price-option**
replication with realistic frictions.

> Research / educational project. Nothing here is investment advice, and the
> backtests are not a live track record. See [Disclaimer](#disclaimer).

---

## 1. Idea in one paragraph

Traders are modelled as spins in a two-state Ising-like system: each agent is either
bullish or bearish, and switches state under the combined influence of the recent
price trend and of the majority opinion. A mean-field approximation reduces the
system to a two-parameter map

$$\langle x \rangle_{t+1} = \tanh\big(a\,\Delta p_t + b\,s_t\big)$$

where $a$ captures trend-chasing and $b$ herding. The parameters are re-estimated at
every date on a rolling window (walk-forward, one step ahead), which makes the model
adaptive to regime shifts. The **residual** between the predicted and the realised
auxiliary series carries a mean-reverting component: when a forecast residual lands
in the tails of its historical distribution, the strategy takes the opposite side.
Positions are expressed as **average-price-option** payoffs, so the strategy is
evaluated on a structure that is actually tradable, slippage and premia included.

## 2. Repository layout

```
.
├── notebooks/
│   ├── ABM_analysis.ipynb              # The model: theory, estimation, benchmark vs. linear regression
│   ├── PCA_residuals_1.6.2_PnL.ipynb   # ← Current version of the strategy (start here)
│   ├── PCA_residuals_1.6.1_PnL.ipynb   # Previous iteration, kept for traceability
│   └── PCA_residuals_1.5_PnL.ipynb     # First complete iteration
├── data/
│   ├── crypto_ohlc.pkl                 # Daily OHLC, ~654 coins, 2019-01-02 → 2025-01-01 (Git LFS)
│   └── returns_BTC.pkl                 # Daily BTC returns, 2019-01-03 → 2023-12-31
├── utils.py                            # Ising/ABM estimator (CPU + PyTorch GPU implementation)
├── requirements.txt
├── .gitattributes                      # Git LFS tracking for *.pkl
└── LICENSE
```

## 3. What each notebook does

### `ABM_analysis.ipynb` — the forecasting engine

Derivation and validation of the model, then a head-to-head comparison against a
rolling linear-regression benchmark along three axes:

1. **Empirical forecasting performance** — R², MAE, RMSE on one-step-ahead forecasts.
   The two models land close, with a slight edge for the ABM.
2. **Adaptation under stress** — a simulated jump-diffusion crash, then the
   autocorrelation function of the prediction errors. The ABM shows a markedly
   shorter relaxation time: the linear model drags its "memory" of the pre-crash
   regime, the non-linear `tanh` map re-anchors much faster.
3. **Signal complexity** — predictive performance as a function of the Shannon
   entropy of the power spectral density, i.e. how structured or noisy the local
   regime is.

### `PCA_residuals_*.ipynb` — the strategy

- **In-sample analysis** — the predicted vs. realised residual scatter shows a
  negative linear relationship with R² ≈ 0.30 on BTC, concentrated in the tails.
  That tail concentration is what the signal exploits.
- **Signal construction** — a position is opened when the forecast residual crosses
  the top/bottom percentile of its historical distribution, and closed when the
  signal flips or returns to zero.
- **PnL accounting** — computed daily, position by position, with the price at entry
  used as the strike and the notional converted into units at that strike.
- **Portfolio** — equal allocation of $1 per coin, each sleeve traded independently,
  so no single asset dominates the aggregate.
- **Out-of-sample** — walk-forward with an expanding in-sample window, so the
  percentile thresholds are refreshed with new observations instead of being fitted
  once on the full history.
- **Synthetic asset construction** — the exposure is replicated with an
  option-based synthetic forward on average-price options, priced in a
  Black–Scholes framework ($S_t$, ATM strike, forward-filled risk-free rate,
  rolling annualised volatility), with an explicit **slippage sensitivity** study
  at opening, early close and settlement.
- **Conclusion** — asset selection and market feasibility.

Version history: `1.5` is the first complete iteration, `1.6.1` and `1.6.2` refine
the methodology and the write-up. **`1.6.2` is the reference version.**

## 4. The estimator (`utils.py`)

| Function | What it does |
| --- | --- |
| `evolution_function(s_t, delta_p, a, b)` | The mean-field map $\tanh(a\,\Delta p + b\,s)$. |
| `energy_function(delta_p, s_t, a, b)` | Sum of squared errors between the realised and predicted normalised series. |
| `two_state_ising_prediction(series, normalization_window, optimisation_window, env_window)` | Walk-forward estimation of $(a_t, b_t)$ on the CPU. Global search with `differential_evolution`, then local polish with `L-BFGS-B`. Returns a DataFrame of `a`, `b`, and the predicted series in normalised and raw units. |
| `two_state_ising_prediction_gpu(...)` | PyTorch implementation that optimises all rolling windows in parallel by gradient descent. Same signature plus `n_steps`, `lr`, `device`. Falls back to CPU when CUDA is unavailable. |

The CPU version runs a full global optimisation per date and is slow on long series;
the GPU version is the one to use for the full cross-section.

## 5. Getting started

```bash
# 1. Git LFS is required: the OHLC dataset (~46 MB) is stored through it
git lfs install
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

# 2. Environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run
jupyter lab
```

Then open `notebooks/PCA_residuals_1.6.2_PnL.ipynb` and run it top to bottom.
Notebooks read the datasets through `../data/`, so run them from the `notebooks/`
directory (which Jupyter does by default).

**If `git lfs` is missing**, install it from [git-lfs.com](https://git-lfs.com) *before*
cloning — otherwise the `.pkl` files arrive as text pointer stubs and
`pd.read_pickle` fails.

## 6. Data

`crypto_ohlc.pkl` is a `DataFrame` with a `(ticker, field)` column MultiIndex —
`open`, `high`, `low`, `close` for ~654 coins — indexed by daily dates from
2019-01-02 to 2025-01-01. Coverage starts at different dates depending on the coin,
so early rows are sparse by construction.

`returns_BTC.pkl` is a daily BTC return `Series` from 2019-01-03 to 2023-12-31, used
by the ABM notebook.

## 7. References

- Kaizoji, T. (2000). *Speculative bubbles and crashes in stock markets: an
  interacting-agent model of speculative activity.* Physica A, 287(3–4), 493–506.
- Kaizoji, T., Bornholdt, S., Fujiwara, Y. (2006). *Dynamics of price and trading
  volume in a spin model of stock markets with heterogeneous agents.* Physica A.

## 8. Disclaimer

This repository is a research and educational project. Backtested results are
hypothetical, depend on the modelling and friction assumptions documented in the
notebooks, and do not represent actual trading. Nothing here constitutes investment
advice.

## 9. License

[MIT](LICENSE)
