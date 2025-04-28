import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, AroonIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from .config import TP_PIPS, SL_PIPS, SPREAD_PIPS, LABEL_MAX_MINUTES, load_selected_features

# ===== 追加: 計算可能な全候補特徴量リスト =============================
_ALL_FEATURES = [
    'rsi', 'rsi_trend',
    'adx',
    'atr', 'atr_change',
    'obv',
    'aroon_up', 'aroon_down',
    'price_pos', 'range',
    'vol_roc5', 'vol_roc10',
]

def list_available_features() -> list:
    """
    このモジュールで計算可能な全ての特徴量名を返す。
    eval_features.py で自動取得するためのヘルパ。
    """
    return _ALL_FEATURES.copy()

# ===== デフォルト選択（fallback 用） ==================================
_DEFAULT_FEATURES = [
    'rsi_trend', 'adx', 'atr_change', 'aroon_down', 'aroon_up',
    'obv', 'price_pos', 'range', 'vol_roc5', 'vol_roc10'
]

def _get_selected_features():
    sel = load_selected_features()
    return sel if sel else _DEFAULT_FEATURES

def compute_features_and_labels(df: pd.DataFrame, selected_features: list | None = None) -> pd.DataFrame:
    """
    ・selected_features が None のときは config から読み込む
    ・指定された特徴量だけを計算
    ・未選択列は最後に削除
    """
    if selected_features is None:
        selected_features = _get_selected_features()

    # --- モメンタム・トレンド・ボラティリティ ------------------
    if any(f in selected_features for f in ["rsi", "rsi_trend"]):
        df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
        if "rsi_trend" in selected_features:
            df["rsi_trend"] = df["rsi"].diff(5)

    if "adx" in selected_features:
        df["adx"] = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx()

    if any(f in selected_features for f in ["atr", "atr_change"]):
        df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
        if "atr_change" in selected_features:
            df["atr_change"] = df["atr"].pct_change()

    # --- OBV -----------------------------------------------------
    if "obv" in selected_features:
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # --- Aroon ---------------------------------------------------
    if any(f in selected_features for f in ["aroon_up", "aroon_down"]):
        aroon = AroonIndicator(high=df["high"], low=df["low"], window=14)
        if "aroon_up" in selected_features:
            df["aroon_up"]   = aroon.aroon_up()
        if "aroon_down" in selected_features:
            df["aroon_down"] = aroon.aroon_down()

    # --- プライスポジション／レンジ ------------------------------
    if "price_pos" in selected_features:
        df["price_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    if "range" in selected_features:
        df["range"] = df["high"] - df["low"]

    # --- ボリューム変化率 ---------------------------------------
    if "vol_roc5" in selected_features:
        df["vol_roc5"]  = df["volume"].pct_change(5)
    if "vol_roc10" in selected_features:
        df["vol_roc10"] = df["volume"].pct_change(10)

    # ===== ラベル付け（既存ロジック） =============================
    n = len(df)
    label_buy  = np.full(n, np.nan)
    label_sell = np.full(n, np.nan)
    time_buy   = np.full(n, np.nan)
    time_sell  = np.full(n, np.nan)
    half_spread = (SPREAD_PIPS / 2) * 0.01

    for i in range(n - LABEL_MAX_MINUTES):
        mid     = df["close"].iat[i]
        entry_b = mid + half_spread
        entry_s = mid - half_spread
        tp_b    = entry_b + TP_PIPS * 0.01
        sl_b    = entry_b - SL_PIPS * 0.01
        tp_s    = entry_s - TP_PIPS * 0.01
        sl_s    = entry_s + SL_PIPS * 0.01

        skip = False
        for j in range(i + 1, i + 1 + LABEL_MAX_MINUTES):
            h, l = df["high"].iat[j], df["low"].iat[j]
            if (h >= tp_b and l <= sl_b) or (l <= tp_s and h >= sl_s):
                skip = True
                break
            if np.isnan(label_buy[i]):
                if h >= tp_b:
                    label_buy[i], time_buy[i] = 1, j - i
                elif l <= sl_b:
                    label_buy[i], time_buy[i] = 0, j - i
            if np.isnan(label_sell[i]):
                if l <= tp_s:
                    label_sell[i], time_sell[i] = 1, j - i
                elif h >= sl_s:
                    label_sell[i], time_sell[i] = 0, j - i
            if not np.isnan(label_buy[i]) and not np.isnan(label_sell[i]):
                break
        if skip:
            label_buy[i] = label_sell[i] = time_buy[i] = time_sell[i] = np.nan

    df["label_buy"]  = label_buy
    df["label_sell"] = label_sell
    df["time_buy"]   = time_buy
    df["time_sell"]  = time_sell

    # --- 未選択列を削除 -----------------------------------------
    drop_cols = [c for c in df.columns
                 if c not in selected_features
                 and c not in ["label_buy", "label_sell", "time_buy", "time_sell",
                               "open", "high", "low", "close", "volume"]]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df
