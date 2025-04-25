import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, AroonIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from .config import TP_PIPS, SL_PIPS, SPREAD_PIPS


def compute_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    # モメンタム・トレンド・ボラティリティ
    df["rsi"]        = RSIIndicator(close=df["close"], window=14).rsi()
    df["rsi_trend"]  = df["rsi"].diff(5)
    df["adx"]        = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx()
    df["atr"]        = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    df["atr_change"] = df["atr"].pct_change()

    # オンバランスボリューム
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # Aroon
    aroon = AroonIndicator(high=df["high"], low=df["low"], window=14)
    df["aroon_up"]   = aroon.aroon_up()
    df["aroon_down"] = aroon.aroon_down()

    # プライスポジション・レンジ
    df["price_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    df["range"]     = df["high"] - df["low"]

    # ボリューム変化率
    df["vol_roc5"]  = df["volume"].pct_change(5)
    df["vol_roc10"] = df["volume"].pct_change(10)

    # ラベル付け（30分先のTP/SL判定） + 到達までの分数
    n = len(df)
    label_buy  = np.full(n, np.nan)
    label_sell = np.full(n, np.nan)
    time_buy   = np.full(n, np.nan)
    time_sell  = np.full(n, np.nan)
    half_spread = (SPREAD_PIPS / 2) * 0.01

    for i in range(n - 30):
        mid     = df["close"].iat[i]
        entry_b = mid + half_spread
        entry_s = mid - half_spread
        tp_b    = entry_b + TP_PIPS * 0.01
        sl_b    = entry_b - SL_PIPS * 0.01
        tp_s    = entry_s - TP_PIPS * 0.01
        sl_s    = entry_s + SL_PIPS * 0.01

        skip = False
        for j in range(i+1, i+31):
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

    return df