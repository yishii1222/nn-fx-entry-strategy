import numpy as np
import pandas as pd
from ta.trend import (
    ADXIndicator, AroonIndicator, EMAIndicator, SMAIndicator, MACD,
    CCIIndicator                       # ★ 追加: CCI 用
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
)
from ta.volatility import (
    AverageTrueRange, BollingerBands, KeltnerChannel
)
from ta.volume import (
    MFIIndicator, ChaikinMoneyFlowIndicator
)
from .config import TP_PIPS, SL_PIPS, SPREAD_PIPS, LABEL_MAX_MINUTES, load_selected_features

# ===== 追加: 計算可能な全候補特徴量リスト =============================
_ALL_FEATURES = [
    # 既存 -----------------------------
    'rsi', 'rsi_trend',
    'adx',
    'atr', 'atr_change',
    'obv',
    'aroon_up', 'aroon_down',
    'price_pos', 'range',
    'vol_roc5', 'vol_roc10',
    # 新規: 移動平均距離 ---------------
    'dist_to_sma_5', 'dist_to_sma_20', 'dist_to_sma_40',
    'dist_to_sma_240', 'dist_to_sma_1000',
    'dist_to_ema_5', 'dist_to_ema_20', 'dist_to_ema_40',
    'dist_to_ema_240', 'dist_to_ema_1000',
    # 新規: ボリンジャーバンド ---------
    'bb_upper', 'bb_lower', 'bb_width', 'bb_percent',
    # 新規: ケルトナーチャンネル -------
    'kc_upper', 'kc_lower', 'kc_width', 'kc_percent',
    # 新規: MACD -----------------------
    'macd', 'macd_signal', 'macd_hist',
    # 新規: ストキャスティクス ----------
    'stoch_k', 'stoch_d',
    # 新規: CCI / Williams %R ----------
    'cci', 'willr',
    # 新規: MFI / CMF ------------------
    'mfi', 'cmf',
    # 新規: ROC / モメンタム ------------
    'roc_1', 'roc_5', 'momentum_5',
    # 新規: リターン --------------------
    'ret_1', 'ret_5',
    # 新規: 高低差比率 ------------------
    'hl_ratio'
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

    # ======== ここから新規特徴量群 =================================

    # --- 移動平均距離 (SMA/EMA) ----------------------------------
    sma_map = {
        5: "dist_to_sma_5", 20: "dist_to_sma_20",
        40: "dist_to_sma_40", 240: "dist_to_sma_240",
        1000: "dist_to_sma_1000"
    }
    if any(name in selected_features for name in sma_map.values()):
        for w, cname in sma_map.items():
            if cname in selected_features:
                sma = SMAIndicator(close=df["close"], window=w).sma_indicator()
                df[cname] = (df["close"] - sma) / sma

    ema_map = {
        5: "dist_to_ema_5", 20: "dist_to_ema_20",
        40: "dist_to_ema_40", 240: "dist_to_ema_240",
        1000: "dist_to_ema_1000"
    }
    if any(name in selected_features for name in ema_map.values()):
        for w, cname in ema_map.items():
            if cname in selected_features:
                ema = EMAIndicator(close=df["close"], window=w).ema_indicator()
                df[cname] = (df["close"] - ema) / ema

    # --- ボリンジャーバンド -------------------------------------
    if any(f in selected_features for f in ["bb_upper", "bb_lower", "bb_width", "bb_percent"]):
        bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        if "bb_upper" in selected_features:
            df["bb_upper"] = bb.bollinger_hband()
        if "bb_lower" in selected_features:
            df["bb_lower"] = bb.bollinger_lband()
        if "bb_width" in selected_features:
            df["bb_width"] = bb.bollinger_wband()
        if "bb_percent" in selected_features:
            df["bb_percent"] = bb.bollinger_pband()

    # --- ケルトナーチャンネル -----------------------------------
    if any(f in selected_features for f in ["kc_upper", "kc_lower", "kc_width", "kc_percent"]):
        kc = KeltnerChannel(high=df["high"], low=df["low"], close=df["close"], window=20)
        if "kc_upper" in selected_features:
            df["kc_upper"] = kc.keltner_channel_hband()
        if "kc_lower" in selected_features:
            df["kc_lower"] = kc.keltner_channel_lband()
        if "kc_width" in selected_features:
            df["kc_width"] = kc.keltner_channel_wband()
        if "kc_percent" in selected_features:
            df["kc_percent"] = kc.keltner_channel_pband()

    # --- MACD ----------------------------------------------------
    if any(f in selected_features for f in ["macd", "macd_signal", "macd_hist"]):
        macd_ind = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        if "macd" in selected_features:
            df["macd"] = macd_ind.macd()
        if "macd_signal" in selected_features:
            df["macd_signal"] = macd_ind.macd_signal()
        if "macd_hist" in selected_features:
            df["macd_hist"] = macd_ind.macd_diff()

    # --- ストキャスティクス --------------------------------------
    if any(f in selected_features for f in ["stoch_k", "stoch_d"]):
        stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
        if "stoch_k" in selected_features:
            df["stoch_k"] = stoch.stoch()
        if "stoch_d" in selected_features:
            df["stoch_d"] = stoch.stoch_signal()

    # --- CCI / Williams %R --------------------------------------
    if "cci" in selected_features:
        df["cci"] = CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=20).cci()
    if "willr" in selected_features:
        df["willr"] = WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=14).williams_r()

    # --- MFI / CMF ----------------------------------------------
    if "mfi" in selected_features:
        df["mfi"] = MFIIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14).money_flow_index()
    if "cmf" in selected_features:
        df["cmf"] = ChaikinMoneyFlowIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=20).chaikin_money_flow()

    # --- ROC / モメンタム ---------------------------------------
    if any(f in selected_features for f in ["roc_1", "roc_5"]):
        if "roc_1" in selected_features:
            df["roc_1"] = ROCIndicator(close=df["close"], window=1).roc()
        if "roc_5" in selected_features:
            df["roc_5"] = ROCIndicator(close=df["close"], window=5).roc()
    if "momentum_5" in selected_features:
        df["momentum_5"] = df["close"].diff(5)

    # --- リターン -----------------------------------------------
    if "ret_1" in selected_features:
        df["ret_1"] = df["close"].pct_change(1)
    if "ret_5" in selected_features:
        df["ret_5"] = df["close"].pct_change(5)

    # --- 高低差比率 ---------------------------------------------
    if "hl_ratio" in selected_features:
        df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]

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
