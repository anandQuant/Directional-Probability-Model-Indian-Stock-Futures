# Directional-Probability-Model-Indian-Stock-Futures
A machine learning model that predicts the probability of price moving Up / Down / Sideways over the next N candles on Indian single stock futures (NSE). Built as part of an MFE research project.

What it does
For any given candle, the model outputs three calibrated probabilities:
P(Up)       = 44.2%  ███████████████
P(Down)     = 39.2%  █████████████
P(Sideways) = 16.6%  █████
These are real probabilities — not raw scores. A prediction of P(Up) = 0.70 means price went up roughly 70% of the time historically when the model said that.

Results (Reliance Industries, 5-min, 2020–2026)
ModelAccuracyMacro F1Log LossLogistic Regression38%0.381.084Random Forest39%0.391.081LightGBM39%0.371.099

Train period: 2020–2024
Test period: 2025–2026 (completely unseen during training)
Random baseline: 33% (3-class problem)
The model consistently beats random by 5–6 percentage points on unseen data


Features (50 total)
The model uses five groups of technical features engineered from raw OHLCV data:
Returns — price change over 1, 3, 6, 12 candles
Volatility — realized vol (6/12/24 window), ATR normalized by price, vol expansion/contraction ratio
Volume — volume surprise vs 24-period average, OBV normalized, up/down volume flow ratio, CMF (Chaikin Money Flow), volume spike z-score
Trend indicators — EMA 9/21/50/200 crossover gaps, price position vs each EMA, EMA alignment score, EMA slope, Supertrend (fast + slow), VWAP distance (daily reset), VWAP crossover events
Momentum & structure — RSI 7 and 14, RSI crossovers, candle body ratio, close position within range
Session features — NSE opening 30 min flag, closing 30 min flag, morning session flag, normalized hour of day

Key design decisions
Adaptive threshold — instead of a hardcoded return threshold for Up/Down labels, the model computes threshold = median(|future_return|) × 0.5 from the data itself. This guarantees Up + Down together are always ~50% of rows regardless of the stock or timeframe. A hardcoded threshold on 5-min data produces 93%+ Sideways labels and a model that never predicts direction.
Adaptive horizon — the prediction horizon (how many candles ahead) is also computed from the data. If the stock's median absolute move over 12 candles is below 0.1%, the horizon extends to 18.
Year-based train/test split — never random split. Train on 2020–2024, test on 2025–2026. No data leakage across the boundary.
Calibrated probabilities — Logistic Regression and Random Forest are wrapped in CalibratedClassifierCV (isotonic regression). Raw model scores are not probabilities. Calibration ensures P(Up) = 0.7 corresponds to ~70% actual frequency.
Daily VWAP reset — VWAP resets each trading day, not cumulative across years. Cumulative VWAP over 6 years is meaningless as an institutional benchmark.
Class balance — class_weight="balanced" on all models. LightGBM has native support. GradientBoosting (fallback) uses compute_sample_weight.

Project structure
├── direction_probability_model.py   # full model — load, label, features, train, eval
├── README.md
├── probability_chart.png            # generated on run — price + ST + probabilities + labels
└── feature_importance.png           # generated on run — top 20 RF feature importances

Requirements
bashpip install pandas numpy scikit-learn matplotlib lightgbm
LightGBM is recommended (30 sec training). If not installed, the model automatically falls back to scikit-learn's GradientBoostingClassifier (~15 min).
Python 3.8+

Data format
One CSV file containing all years. Columns:
date, open, high, low, close, volume
2020-01-01 09:15:00, 716.7, 719.9, 716.7, 718.7, 561250
2020-01-01 09:20:00, 718.8, 721.0, 718.7, 720.1, 514278

Timestamps in IST (no timezone offset needed)
5-min candle interval
Tested on Reliance Industries (RELIANCE.NS) futures data

If your column is named timestamp instead of date, the loader handles that automatically.

How to run
1. Clone the repo and install dependencies
bashgit clone https://github.com/yourusername/direction-probability-model.git
cd direction-probability-model
pip install pandas numpy scikit-learn matplotlib lightgbm
2. Add your data file
Place your CSV in the project folder. Update the path at the bottom of direction_probability_model.py:
pythonCSV_PATH = r"your_5min_data_2020_2026.csv"
3. Run
bashpython direction_probability_model.py
4. Output
STEP 1 — Loading data
✅ Loaded 287,432 rows | 2020-01-01 → 2026-01-19

STEP 2 — Creating labels (adaptive horizon + threshold)
📐 Adaptive horizon   : 12 candles (60 min ahead)
   Adaptive threshold : 0.00062  (0.0620%)
📊 Label distribution:
   Up         95,812  (33.3%)  █████████████
   Down       96,104  (33.4%)  █████████████
   Sideways   95,516  (33.2%)  █████████████

...

🎯 LightGBM — next 12 candles (60 min):
   Candle : 2026-01-19 14:25:00
   Price  : 1411.90
   P(Up       ) = 46.87%  ████████████████
   P(Down     ) = 34.07%  ███████████
   P(Sideways ) = 19.06%  ██████

How to interpret the signal
The model outputs probabilities on every candle. Not every prediction is tradeable.
Use the consensus rule: only act when at least 2 of 3 models agree on the same direction AND the winning probability exceeds 50%.
SignalInterpretationAll 3 models > 55% UpStrong Up signal2 of 3 models > 50% UpModerate Up signalModels disagreeNo trade — uncertainAll probabilities ~33%Market is in noise — avoid
This model does not include transaction costs. A 5–6% edge over random can easily be wiped out by brokerage, slippage, and impact cost on futures. Always backtest with realistic costs before using in live trading.

Limitations

Trained and tested on a single stock (Reliance Industries). Signals may not generalize to other stocks without retraining.
5-min intraday direction is inherently noisy. 38–39% accuracy on a 3-class problem is a real edge but a small one.
The model does not account for fundamental regime changes (new management, sector disruption, macro shocks).
No walk-forward retraining. A static model trained on 2020–2024 may degrade as 2026 market structure evolves.
hour_norm is the top feature (12% importance), meaning the model partially learns session-specific patterns rather than pure direction. This is a real signal but also a limitation — the model knows when large moves happen, not always which way.


Planned improvements

 Walk-forward retraining (rolling 6-month windows)
 Interaction features: session × volatility, OBV × momentum
 Signal confidence filter — only emit predictions when top probability > 0.50
 HTF (weekly/monthly) model using fundamental data — balance sheet, earnings, cash flow, ratios
 Multi-stock version with sector regime features


Academic context
This project was built as part of an MFE (Master of Financial Engineering) program to explore the application of supervised ML to intraday direction prediction on Indian equity futures. The focus is on correct methodology — adaptive labelling, proper time-series validation, calibrated probabilities — rather than maximizing backtest performance.
Key references:

Lopez de Prado, M. — Advances in Financial Machine Learning (label construction, walk-forward validation)
Platt, J. — Probabilistic outputs for SVMs (calibration)
Brier, G.W. — Verification of forecasts (Brier score for probability evaluation)


License
MIT — free to use, modify, and distribute. If you use this in research, a citation or acknowledgement is appreciated.
