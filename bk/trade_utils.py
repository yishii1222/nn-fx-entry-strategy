import os
import json
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator, AroonIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import requests
from dateutil import parser
import time
from datetime import timedelta, timezone
from scipy.stats import binomtest

# ====== 可変設定値 ======
TP_PIPS             = 10               # Before : 5
SL_PIPS             = 10               # Before : 3
DAYS_BACK           = 10               # Before : 5
K_RATIO             = 0.01             # Before : 0.05
DIST_MEAN_THRESH    = 1.0
DIST_MAX_THRESH     = 2.0
EPSILON             = 1e-6
SIGNIFICANCE_LEVEL  = 0.05

# ====== 固定設定値 ======
SPREAD_PIPS         = 0.2
ACCESS_TOKEN        = '0277ce3e38b48230acfb6ed493c43a70-c4952ef6136325c9f5c713d1d218fdaa'
INSTRUMENT          = 'USD_JPY'
JST                 = timezone(timedelta(hours=9))
THRESH_PATH         = "optuna_config.json"

# Optuna時に設定値取得
def load_thresholds_from_file():
    global DIST_MEAN_THRESH, DIST_MAX_THRESH
    if os.path.exists(THRESH_PATH):
        with open(THRESH_PATH, "r") as f:
            config = json.load(f)
            DIST_MEAN_THRESH = config.get("mean_thresh", DIST_MEAN_THRESH)
            DIST_MAX_THRESH  = config.get("max_thresh", DIST_MAX_THRESH)


def fetch_1min_data(start: pd.Timestamp, end: pd.Timestamp, access_token: str, instrument: str) -> pd.DataFrame:
    url     = f'https://api-fxtrade.oanda.com/v3/instruments/{instrument}/candles'
    headers = {'Authorization': f'Bearer {access_token}'}
    current = start
    all_data = []
    while current < end:
        params = {
            'from':        current.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'granularity': 'M1',
            'count':       500,
            'price':       'M'
        }
        try:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json().get('candles', [])
        except Exception as e:
            print(f"Fetch error: {e}")
            break

        valid = [c for c in data if c.get('complete')]
        if not valid:
            current += timedelta(minutes=1)
            continue

        for c in valid:
            all_data.append({
                'time':   c['time'],
                'open':   float(c['mid']['o']),
                'high':   float(c['mid']['h']),
                'low':    float(c['mid']['l']),
                'close':  float(c['mid']['c']),
                'volume': int(c['volume'])
            })

        current = parser.isoparse(valid[-1]['time']) + timedelta(minutes=1)
        time.sleep(0.2)

    df = pd.DataFrame(all_data)
    if df.empty:
        return df
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)
    return df


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


def estimate_signals(train_df: pd.DataFrame, latest_row: pd.Series) -> tuple:
    feat_cols = [
        'rsi_trend', 'adx', 'atr_change', 'aroon_down', 'aroon_up',
        'obv', 'price_pos', 'range', 'vol_roc5', 'vol_roc10'
    ]
    mask_all  = train_df[feat_cols].notna().all(axis=1)

    # 最新のデータに欠損値が含まれる場合はシグナル判定をスキップ
    if not latest_row[feat_cols].notna().all():
        return False, False, {}

    mb_tr = mask_all & train_df["label_buy"].notna()
    ms_tr = mask_all & train_df["label_sell"].notna()

    Xb_tr, yb_tr = train_df.loc[mb_tr, feat_cols].values, train_df.loc[mb_tr, "label_buy"].to_numpy()
    Xs_tr, ys_tr = train_df.loc[ms_tr, feat_cols].values, train_df.loc[ms_tr, "label_sell"].to_numpy()
    if len(Xb_tr) < 1 or len(Xs_tr) < 1:
        return False, False, {}

    scaler = StandardScaler()
    scaler.fit(train_df.loc[mb_tr|ms_tr, feat_cols].values)

    Xb_sc = scaler.transform(Xb_tr)
    Xs_sc = scaler.transform(Xs_tr)
    cur_sc = scaler.transform([[latest_row[c] for c in feat_cols]])

    k_nb_buy = max(1, int(len(Xb_tr) * K_RATIO))
    nbr_b    = NearestNeighbors(n_neighbors=k_nb_buy, n_jobs=-1).fit(Xb_sc)
    db, ib   = nbr_b.kneighbors(cur_sc, return_distance=True)
    db, ib   = db[0], ib[0]
    buy_rate = ((1.0/(db+EPSILON)) * yb_tr[ib]).sum() / (1.0/(db+EPSILON)).sum() * 100
    mean_db, max_db = db.mean(), db.max()

    k_nb_sell = max(1, int(len(Xs_tr) * K_RATIO))
    nbr_s     = NearestNeighbors(n_neighbors=k_nb_sell, n_jobs=-1).fit(Xs_sc)
    ds, is_   = nbr_s.kneighbors(cur_sc, return_distance=True)
    ds, is_   = ds[0], is_[0]
    sell_rate = ((1.0/(ds+EPSILON)) * ys_tr[is_]).sum() / (1.0/(ds+EPSILON)).sum() * 100
    mean_ds, max_ds = ds.mean(), ds.max()

    threshold = SL_PIPS / (TP_PIPS + SL_PIPS) * 100

    # 統計的有意性検定
    wins_b = int(yb_tr[ib].sum())
    wins_s = int(ys_tr[is_].sum())
    res_b = binomtest(wins_b, n=k_nb_buy, p=threshold/100, alternative='greater')
    res_s = binomtest(wins_s, n=k_nb_sell, p=threshold/100, alternative='greater')
    pval_b = res_b.pvalue
    pval_s = res_s.pvalue
    sig_b  = (pval_b < SIGNIFICANCE_LEVEL)
    sig_s  = (pval_s < SIGNIFICANCE_LEVEL)

    metrics = {
        'threshold':     threshold,
        'buy_rate':      buy_rate,
        'sell_rate':     sell_rate,
        'mean_db':       mean_db,
        'max_db':        max_db,
        'mean_ds':       mean_ds,
        'max_ds':        max_ds,
        'mean_d_thresh': DIST_MEAN_THRESH,
        'max_d_thresh':  DIST_MAX_THRESH,
        'n_k_nb_buy':    k_nb_buy,
        'n_k_nb_sell':   k_nb_sell,
        'pval_buy':      pval_b,
        'pval_sell':     pval_s,
        'sig_buy':       sig_b,
        'sig_sell':      sig_s,
        'significance_level': SIGNIFICANCE_LEVEL,
    }

    buy_ok  = (
        buy_rate >= threshold and
        mean_db <= DIST_MEAN_THRESH and
        max_db  <= DIST_MAX_THRESH and
        sig_b
    )
    sell_ok = (
        sell_rate >= threshold and
        mean_ds <= DIST_MEAN_THRESH and
        max_ds  <= DIST_MAX_THRESH and
        sig_s
    )

    return buy_ok, sell_ok, metrics


def explain_reason(ok_flag, label, metrics, prefix):
    reasons = []
    if not ok_flag:
        if metrics.get(f"{label}_rate", 0) < metrics.get('threshold', 0):
            reasons.append(f"{label}_rate < threshold")
        if metrics.get(f"mean_d{label[0]}", np.inf) > metrics.get('mean_d_thresh', np.inf):
            reasons.append(f"mean_distance > {metrics['mean_d_thresh']}")
        if metrics.get(f"max_d{label[0]}", np.inf) > metrics.get('max_d_thresh', np.inf):
            reasons.append(f"max_distance > {metrics['max_d_thresh']}")
        if label == 'buy' and not metrics.get('sig_buy', False):
            reasons.append("buy not statistically significant")
        if label == 'sell' and not metrics.get('sig_sell', False):
            reasons.append("sell not statistically significant")
    if reasons:
        print(f"→ {label.upper()} 非成立理由: {', '.join(reasons)}")
