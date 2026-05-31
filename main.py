# =============================================================================
# DIRECTIONAL PROBABILITY MODEL — Single Stock Futures
# Data  : 5-min OHLCV, 2020–2026 in one CSV
# Train : 2020–2024  |  Test : 2025–2026
# Predicts P(Up) / P(Down) / P(Sideways) over next N candles
# Columns: date, open, high, low, close, volume
#
# HOW TO RUN:
#   pip install pandas numpy scikit-learn matplotlib lightgbm
#   python direction_probability_model.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("⚠️  LightGBM not found. Run: pip install lightgbm")
    print("   Falling back to GradientBoostingClassifier (slower)\n")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight

# =============================================================================
# STEP 1 — LOAD CSV
# =============================================================================
# Handles your exact format:
#   date, open, high, low, close, volume
#   2020-01-01 09:15:00  (no timezone)

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    # Your column is 'date' — rename to 'datetime' for consistency
    if "date" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"date": "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"])   # no utc needed
    df = df.sort_values("datetime").reset_index(drop=True)

    years = sorted(df["datetime"].dt.year.unique())
    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Period : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"   Years  : {years}")
    print(f"   Candle : 5-min")
    return df

# =============================================================================
# STEP 2 — LABELS  (adaptive horizon + adaptive threshold)
# =============================================================================
# Both horizon and threshold are computed FROM YOUR DATA so they
# always produce a balanced label distribution regardless of the stock.
#
# Adaptive horizon:
#   We want to predict ~60 min ahead on 5-min data = 12 candles.
#   But we also check: does the median |return| over 12 candles give
#   enough signal? If the stock barely moves, we extend to 18.
#   Rule: use 12 candles; if median |future_return| < 0.1%, use 18.
#
# Adaptive threshold:
#   threshold = median(|future_return|) × 0.5
#   Guarantees Up + Down ≈ 50% of all rows.

def create_labels(df):
    df = df.copy()

    # --- Decide horizon ---
    # Try 12 candles (60 min) first
    df["_fr12"] = (df["close"].shift(-12) - df["close"]) / df["close"]
    median_12   = df["_fr12"].abs().dropna().median()

    if median_12 >= 0.001:          # median move ≥ 0.1%  → 12 candles is fine
        HORIZON = 12
    else:                           # stock moves very little → extend to 18
        HORIZON = 18

    df = df.drop(columns=["_fr12"])

    print(f"\n📐 Adaptive horizon   : {HORIZON} candles ({HORIZON * 5} min ahead)")

    df["future_close"]  = df["close"].shift(-HORIZON)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
    df = df.dropna(subset=["future_return"])

    # --- Adaptive threshold ---
    threshold = df["future_return"].abs().median() * 0.5
    print(f"   Adaptive threshold : {threshold:.5f}  ({threshold*100:.4f}%)")
    print(f"   Median |return|    : {df['future_return'].abs().median()*100:.4f}%")

    def label(r):
        if   r >  threshold: return "Up"
        elif r < -threshold: return "Down"
        else:                return "Sideways"

    df["label"] = df["future_return"].apply(label)

    counts = df["label"].value_counts()
    total  = len(df)
    print(f"\n📊 Label distribution (full dataset):")
    for lbl, cnt in counts.items():
        bar = "█" * int(cnt / total * 40)
        print(f"   {lbl:<10} {cnt:>8,}  ({cnt/total:.1%})  {bar}")

    # store horizon for later use in plot title
    df.attrs["horizon"] = HORIZON
    return df, HORIZON

# =============================================================================
# STEP 3 — FEATURE ENGINEERING HELPERS
# =============================================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def supertrend(df, period=10, multiplier=3.0):
    """
    +1 = uptrend (price above band), -1 = downtrend (price below band).
    Flips only when price crosses the band — filters small wiggles.
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(close)
    idx   = df.index

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]

    atr = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr[i] = tr[i - period + 1: i + 1].mean()

    hl2         = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    signal      = np.zeros(n, dtype=float)

    for i in range(1, n):
        if np.isnan(atr[i]) or np.isnan(atr[i - 1]):
            continue
        final_upper[i] = basic_upper[i] if (basic_upper[i] < final_upper[i-1]
                          or close[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = basic_lower[i] if (basic_lower[i] > final_lower[i-1]
                          or close[i-1] < final_lower[i-1]) else final_lower[i-1]
        if   close[i] > final_upper[i-1]: signal[i] =  1.0
        elif close[i] < final_lower[i-1]: signal[i] = -1.0
        else:                              signal[i] =  signal[i-1]

    distance = np.where(signal == 1,
                        (close - final_lower) / (close + 1e-9),
                        (close - final_upper) / (close + 1e-9))
    signal[:period]   = np.nan
    distance[:period] = np.nan
    return pd.Series(signal, index=idx), pd.Series(distance, index=idx)

def vwap_daily(df):
    """
    VWAP reset each trading day — institutionally correct.
    Cumulative VWAP across years loses meaning; daily VWAP is the benchmark.
    """
    df = df.copy()
    df["_date"]    = df["datetime"].dt.date
    tp             = (df["high"] + df["low"] + df["close"]) / 3.0
    df["_tpv"]     = tp * df["volume"]
    df["_cum_tpv"] = df.groupby("_date")["_tpv"].cumsum()
    df["_cum_vol"] = df.groupby("_date")["volume"].cumsum()
    vwap           = df["_cum_tpv"] / (df["_cum_vol"] + 1e-9)
    return vwap

# =============================================================================
# STEP 4 — FEATURE COLUMNS
# =============================================================================

FEATURE_COLS = [
    # Returns
    "ret_1", "ret_3", "ret_6", "ret_12",
    # Volatility
    "vol_6", "vol_12", "vol_24", "vol_ratio", "atr_norm",
    # Volume
    "vol_surprise", "obv_norm",
    "vol_trend", "vol_flow_signal", "vol_spike", "cmf_14",
    "vol_price_efficiency",
    # Momentum & candle shape
    "momentum_6", "momentum_12",
    "body_ratio", "close_position",
    # EMA
    "ema_9_21_diff", "ema_21_50_diff", "ema_50_200_diff",
    "price_vs_ema9", "price_vs_ema21", "price_vs_ema50", "price_vs_ema200",
    "ema_alignment", "ema21_slope", "ema50_slope",
    # Supertrend
    "st_signal_fast", "st_signal_slow",
    "st_dist_fast",   "st_dist_slow",
    "st_agreement",   "st_fast_flip", "st_slow_flip",
    # VWAP (daily reset)
    "price_vs_vwap", "vwap_cross_up", "vwap_cross_down",
    # RSI
    "rsi_14", "rsi_7", "rsi_14_diff", "rsi_7_diff",
    "rsi_cross_up", "rsi_cross_down",
    # Session (NSE 5-min in IST — no UTC shift needed)
    "is_morning_session", "is_open_30min", "is_close_30min", "hour_norm",
]

# =============================================================================
# STEP 5 — BUILD FEATURES
# =============================================================================
# Lookback windows are scaled for 5-min candles:
#   1-min code used windows of 5,10,20 → 5-min uses 6,12,24
#   (roughly same real-time coverage: 30 min, 60 min, 120 min)

def add_features(df):
    df = df.copy()
    c  = df["close"]
    v  = df["volume"]
    h  = df["high"]
    l  = df["low"]

    # ── Returns ──────────────────────────────────────────────────────────────
    # 5-min candles: ret_1=5min, ret_3=15min, ret_6=30min, ret_12=60min
    df["ret_1"]  = c.pct_change(1)
    df["ret_3"]  = c.pct_change(3)
    df["ret_6"]  = c.pct_change(6)
    df["ret_12"] = c.pct_change(12)

    # ── Realized Volatility ───────────────────────────────────────────────────
    # vol_6=30min window, vol_12=60min, vol_24=120min
    df["vol_6"]     = df["ret_1"].rolling(6).std()
    df["vol_12"]    = df["ret_1"].rolling(12).std()
    df["vol_24"]    = df["ret_1"].rolling(24).std()
    df["vol_ratio"] = df["vol_6"] / (df["vol_24"] + 1e-9)

    # ── ATR ───────────────────────────────────────────────────────────────────
    tr             = pd.concat([h - l,
                                (h - c.shift(1)).abs(),
                                (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr_12"]   = tr.rolling(12).mean()
    df["atr_norm"] = df["atr_12"] / c

    # ── Volume ────────────────────────────────────────────────────────────────
    df["vol_ma_24"]    = v.rolling(24).mean()
    df["vol_surprise"] = v / (df["vol_ma_24"] + 1e-9)

    vol_ma_12  = v.rolling(12).mean()
    vol_std    = v.rolling(24).std()
    df["vol_spike"] = (v - df["vol_ma_24"]) / (vol_std + 1e-9)
    df["vol_trend"] = vol_ma_12 / (df["vol_ma_24"] + 1e-9)

    # ── OBV ───────────────────────────────────────────────────────────────────
    direction      = np.sign(df["ret_1"])
    df["obv"]      = (direction * v).cumsum()
    df["obv_norm"] = df["obv"] / (df["obv"].rolling(24).std() + 1e-9)

    # ── Up/Down volume flow ───────────────────────────────────────────────────
    up_vol   = pd.Series(np.where(df["ret_1"] > 0, v, 0), index=df.index).rolling(12).mean()
    down_vol = pd.Series(np.where(df["ret_1"] < 0, v, 0), index=df.index).rolling(12).mean()
    vfr      = up_vol / (down_vol + 1e-9)
    df["vol_flow_signal"] = (vfr - 1) / (vfr + 1 + 1e-9)

    # ── Price-Volume efficiency ───────────────────────────────────────────────
    df["vol_price_efficiency"] = df["ret_1"].abs() / (df["vol_surprise"] + 1e-9)

    # ── CMF (Chaikin Money Flow, 14-period) ───────────────────────────────────
    wick         = (h - l) + 1e-9
    clv          = ((c - l) - (h - c)) / wick
    df["cmf_14"] = (clv * v).rolling(14).sum() / (v.rolling(14).sum() + 1e-9)

    # ── Momentum ─────────────────────────────────────────────────────────────
    df["momentum_6"]  = c / c.shift(6)  - 1
    df["momentum_12"] = c / c.shift(12) - 1

    # ── Candle shape ─────────────────────────────────────────────────────────
    body               = (c - df["open"]).abs()
    df["body_ratio"]   = body / wick
    df["close_position"] = (c - l) / wick

    # ── EMA ───────────────────────────────────────────────────────────────────
    # Periods unchanged — EMA reacts to price not time, so same numbers work
    df["ema_9"]   = ema(c, 9)
    df["ema_21"]  = ema(c, 21)
    df["ema_50"]  = ema(c, 50)
    df["ema_200"] = ema(c, 200)

    df["ema_9_21_diff"]   = (df["ema_9"]   - df["ema_21"])  / c
    df["ema_21_50_diff"]  = (df["ema_21"]  - df["ema_50"])  / c
    df["ema_50_200_diff"] = (df["ema_50"]  - df["ema_200"]) / c

    df["price_vs_ema9"]   = (c - df["ema_9"])   / c
    df["price_vs_ema21"]  = (c - df["ema_21"])  / c
    df["price_vs_ema50"]  = (c - df["ema_50"])  / c
    df["price_vs_ema200"] = (c - df["ema_200"]) / c

    df["ema_alignment"] = (
        (df["ema_9"]  > df["ema_21"]).astype(int) +
        (df["ema_21"] > df["ema_50"]).astype(int) +
        (df["ema_50"] > df["ema_200"]).astype(int)
    )
    df["ema_alignment"] = (df["ema_alignment"] - 1.5) / 1.5
    df["ema21_slope"]   = df["ema_21"].pct_change(3)
    df["ema50_slope"]   = df["ema_50"].pct_change(5)

    # ── Supertrend ────────────────────────────────────────────────────────────
    st_sig_f, st_dist_f = supertrend(df, period=7,  multiplier=2.5)
    st_sig_s, st_dist_s = supertrend(df, period=14, multiplier=3.5)

    df["st_signal_fast"] = st_sig_f
    df["st_signal_slow"] = st_sig_s
    df["st_dist_fast"]   = st_dist_f
    df["st_dist_slow"]   = st_dist_s
    df["st_agreement"]   = np.where(st_sig_f == st_sig_s, st_sig_f, 0)
    df["st_fast_flip"]   = (df["st_signal_fast"].diff().abs() > 0).rolling(3).max()
    df["st_slow_flip"]   = (df["st_signal_slow"].diff().abs() > 0).rolling(3).max()

    # ── VWAP (daily reset — institutionally correct) ───────────────────────
    df["vwap"]          = vwap_daily(df)
    df["price_vs_vwap"] = (c - df["vwap"]) / c
    df["vwap_cross_up"]   = ((c > df["vwap"]) & (c.shift(1) <= df["vwap"].shift(1))).astype(int)
    df["vwap_cross_down"] = ((c < df["vwap"]) & (c.shift(1) >= df["vwap"].shift(1))).astype(int)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["rsi_14"]      = calc_rsi(c, 14) / 100.0
    df["rsi_7"]       = calc_rsi(c, 7)  / 100.0
    df["rsi_14_diff"] = df["rsi_14"] - 0.5
    df["rsi_7_diff"]  = df["rsi_7"]  - 0.5
    df["rsi_cross_up"]   = ((df["rsi_7"] > df["rsi_14"]) &
                             (df["rsi_7"].shift(1) <= df["rsi_14"].shift(1))).astype(int)
    df["rsi_cross_down"] = ((df["rsi_7"] < df["rsi_14"]) &
                             (df["rsi_7"].shift(1) >= df["rsi_14"].shift(1))).astype(int)

    # ── Session (IST — your timestamps are already in IST) ────────────────────
    # NSE: 9:15 AM – 3:30 PM IST
    hour   = df["datetime"].dt.hour
    minute = df["datetime"].dt.minute

    df["is_morning_session"] = (
        ((hour == 9)  & (minute >= 15)) | (hour == 10) | (hour == 11)
    ).astype(int)

    df["is_open_30min"] = (
        ((hour == 9) & (minute >= 15)) | ((hour == 9) & (minute <= 45))
    ).astype(int)

    df["is_close_30min"] = (
        ((hour == 15) & (minute <= 30))
    ).astype(int)

    df["hour_norm"] = hour / 23.0

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    nan_counts = df[FEATURE_COLS].isna().sum()
    bad        = nan_counts[nan_counts > 0]
    if len(bad) > 0:
        print(f"\n⚠️  NaN in features (top 5): {bad.head().to_dict()}")

    before = len(df)
    df     = df.dropna(subset=FEATURE_COLS)
    print(f"\n✅ Features done | {before:,} → {len(df):,} rows "
          f"(dropped {before - len(df):,} warmup rows)")
    return df

# =============================================================================
# STEP 6 — TRAIN / TEST SPLIT  (2020–2024 train | 2025–2026 test)
# =============================================================================

TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
TEST_YEARS  = [2025, 2026]

def train_test_split_timeseries(df):
    train = df[df["datetime"].dt.year.isin(TRAIN_YEARS)].copy()
    test  = df[df["datetime"].dt.year.isin(TEST_YEARS)].copy()

    years_in_data = sorted(df["datetime"].dt.year.unique().tolist())

    if len(train) == 0:
        raise ValueError(f"No train data. Years in CSV: {years_in_data}")
    if len(test) == 0:
        raise ValueError(f"No test data. Years in CSV: {years_in_data}")

    print(f"\n✅ Train (2020–2024) : {len(train):,} rows")
    print(f"   {train['datetime'].min().date()} → {train['datetime'].max().date()}")
    print(f"\n✅ Test  (2025–2026) : {len(test):,} rows")
    print(f"   {test['datetime'].min().date()} → {test['datetime'].max().date()}")

    for name, split in [("TRAIN", train), ("TEST", test)]:
        counts = split["label"].value_counts()
        total  = len(split)
        print(f"\n   {name} labels:")
        for lbl, cnt in counts.items():
            bar = "█" * int(cnt / total * 30)
            print(f"   {lbl:<10} {cnt:>8,}  ({cnt/total:.1%})  {bar}")

    return train, test

# =============================================================================
# STEP 7 — TRAIN MODELS
# =============================================================================

def train_models(train):
    X_train = train[FEATURE_COLS]
    y_train = train["label"]

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("\n🔧 Training Logistic Regression...")
    lr_base  = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
    lr_model = CalibratedClassifierCV(lr_base, cv=3, method="isotonic")
    lr_model.fit(X_train_scaled, y_train)
    print("✅ Done")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n🔧 Training Random Forest (~1 min)...")
    rf_base  = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_model = CalibratedClassifierCV(rf_base, cv=3, method="isotonic")
    rf_model.fit(X_train_scaled, y_train)
    print("✅ Done")

    # ── LightGBM (fast) or GradientBoosting (fallback) ───────────────────────
    if HAS_LGBM:
        print("\n🔧 Training LightGBM (~30 sec)...")
        gb_model = LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_name = "LightGBM"
    else:
        print("\n🔧 Training GradientBoosting (~15 min)...")
        from sklearn.utils.class_weight import compute_sample_weight
        sw       = compute_sample_weight("balanced", y_train)
        gb_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8, random_state=42
        )
        gb_model.fit(X_train_scaled, y_train, sample_weight=sw)
        gb_name  = "GradientBoosting"
    print(f"✅ Done")

    return lr_model, rf_model, gb_model, gb_name, scaler

# =============================================================================
# STEP 8 — EVALUATE
# =============================================================================

def evaluate(model, model_name, test, scaler):
    X_test_scaled = scaler.transform(test[FEATURE_COLS])
    y_test        = test["label"]

    proba   = model.predict_proba(X_test_scaled)
    classes = model.classes_
    pred    = model.predict(X_test_scaled)

    print(f"\n{'='*54}")
    print(f"📈 {model_name} — Test (2025–2026)")
    print(f"{'='*54}")
    print(f"Log Loss : {log_loss(y_test, proba, labels=classes):.4f}")
    print(classification_report(y_test, pred))
    return proba, classes, pred

# =============================================================================
# STEP 9 — FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(model, model_name, top_n=20):
    try:
        # Works for RF (calibrated) and LightGBM/GB (direct)
        if hasattr(model, "calibrated_classifiers_"):
            base  = model.calibrated_classifiers_[0].estimator
            imps  = base.feature_importances_
        else:
            imps  = model.feature_importances_

        feat_imp = pd.Series(imps, index=FEATURE_COLS).sort_values(ascending=False)

        plt.figure(figsize=(10, 7))
        feat_imp.head(top_n).plot(kind="barh", color="steelblue")
        plt.title(f"Top {top_n} Features — {model_name}")
        plt.xlabel("Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("✅ Saved: feature_importance.png")

        print(f"\n🔑 Top 10 features:")
        for feat, imp in feat_imp.head(10).items():
            bar = "█" * int(imp * 200)
            print(f"   {feat:<30} {imp:.4f}  {bar}")

    except Exception as e:
        print(f"⚠️  Feature importance unavailable: {e}")

# =============================================================================
# STEP 10 — PLOT RESULTS
# =============================================================================

def plot_results(df_test, proba, classes, model_name, horizon):
    df_p = df_test.copy().reset_index(drop=True)
    for i, cls in enumerate(classes):
        df_p[f"prob_{cls}"] = proba[:, i]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"{model_name} | 5-min | horizon={horizon} candles ({horizon*5} min)",
                 fontsize=13, fontweight="bold")

    # Price + EMAs
    axes[0].plot(df_p["datetime"], df_p["close"],   color="black",  lw=0.6, label="Price")
    axes[0].plot(df_p["datetime"], df_p["ema_9"],   color="cyan",   lw=0.6, label="EMA9",   alpha=0.8)
    axes[0].plot(df_p["datetime"], df_p["ema_21"],  color="blue",   lw=0.6, label="EMA21",  alpha=0.7)
    axes[0].plot(df_p["datetime"], df_p["ema_50"],  color="orange", lw=0.6, label="EMA50",  alpha=0.7)
    axes[0].plot(df_p["datetime"], df_p["ema_200"], color="red",    lw=0.6, label="EMA200", alpha=0.7)
    axes[0].set_title("Price + EMAs")
    axes[0].legend(loc="upper left", fontsize=7)

    # Supertrend
    axes[1].plot(df_p["datetime"], df_p["st_signal_fast"], color="purple", lw=0.6, label="ST Fast (7,2.5)")
    axes[1].plot(df_p["datetime"], df_p["st_signal_slow"], color="brown",  lw=0.6, label="ST Slow (14,3.5)", alpha=0.7)
    axes[1].axhline(0, color="gray", ls="--", lw=0.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_title("Supertrend (+1 uptrend / −1 downtrend)")
    axes[1].legend(loc="upper left", fontsize=7)

    # Probabilities
    colors = {"Up": "green", "Down": "red", "Sideways": "orange"}
    for cls in classes:
        col = f"prob_{cls}"
        if col in df_p.columns:
            axes[2].plot(df_p["datetime"], df_p[col],
                         label=f"P({cls})", color=colors.get(cls, "blue"), lw=0.8)
    axes[2].axhline(0.33, color="gray", ls="--", lw=0.5, label="Baseline 0.33")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Predicted Probabilities")
    axes[2].legend(loc="upper right", fontsize=7)

    # Actual labels
    lmap = {"Up": 1, "Sideways": 0, "Down": -1}
    cmap = {"Up": "green", "Sideways": "orange", "Down": "red"}
    axes[3].bar(df_p["datetime"], df_p["label"].map(lmap),
                color=df_p["label"].map(cmap), width=0.002)
    axes[3].set_yticks([-1, 0, 1])
    axes[3].set_yticklabels(["Down", "Sideways", "Up"])
    axes[3].set_title("Actual Labels")

    plt.tight_layout()
    plt.savefig("probability_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: probability_chart.png")

# =============================================================================
# STEP 11 — LATEST SIGNAL
# =============================================================================

def predict_latest(model, model_name, df, scaler, horizon):
    last        = df[FEATURE_COLS].iloc[-1:]
    proba       = model.predict_proba(scaler.transform(last))[0]
    classes     = model.classes_
    candle_time = df["datetime"].iloc[-1]
    price       = df["close"].iloc[-1]

    print(f"\n🎯 {model_name} — next {horizon} candles ({horizon*5} min):")
    print(f"   Candle : {candle_time}")
    print(f"   Price  : {price:.2f}")
    for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
        bar  = "█" * int(p * 35)
        flag = " ← SIGNAL" if p > 0.50 else ""
        print(f"   P({cls:<9}) = {p:.2%}  {bar}{flag}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # ── SET YOUR FILE PATH ────────────────────────────────────────────────────
    CSV_PATH = r"C:\Users\Pandya Anand\Desktop\probability ML\RELIANCE.csv"
    # ─────────────────────────────────────────────────────────────────────────

    print("=" * 54)
    print("STEP 1 — Loading data")
    print("=" * 54)
    df = load_data(CSV_PATH)

    print("\n" + "=" * 54)
    print("STEP 2 — Creating labels (adaptive horizon + threshold)")
    print("=" * 54)
    df, HORIZON = create_labels(df)

    print("\n" + "=" * 54)
    print("STEP 3-5 — Engineering features")
    print("=" * 54)
    df = add_features(df)

    print(f"\n🔍 Year distribution after feature engineering:")
    print(df["datetime"].dt.year.value_counts().sort_index().to_string())

    print("\n" + "=" * 54)
    print("STEP 6 — Train / Test split")
    print("=" * 54)
    train, test = train_test_split_timeseries(df)

    print("\n" + "=" * 54)
    print("STEP 7 — Training models")
    print("=" * 54)
    lr_model, rf_model, gb_model, gb_name, scaler = train_models(train)

    print("\n" + "=" * 54)
    print("STEP 8 — Evaluation on 2025–2026")
    print("=" * 54)
    lr_proba, classes, _ = evaluate(lr_model, "Logistic Regression", test, scaler)
    rf_proba, classes, _ = evaluate(rf_model, "Random Forest",       test, scaler)
    gb_proba, classes, _ = evaluate(gb_model, gb_name,               test, scaler)

    print("\n" + "=" * 54)
    print("STEP 9 — Feature Importance")
    print("=" * 54)
    plot_feature_importance(rf_model, "Random Forest")

    plot_results(test, gb_proba, classes, gb_name, HORIZON)

    print("\n" + "=" * 54)
    print("STEP 11 — Latest signal")
    print("=" * 54)
    predict_latest(lr_model, "Logistic Regression", df, scaler, HORIZON)
    predict_latest(rf_model, "Random Forest",        df, scaler, HORIZON)
    predict_latest(gb_model, gb_name,                df, scaler, HORIZON)

    print("\n✅ All done! Files: probability_chart.png, feature_importance.png")