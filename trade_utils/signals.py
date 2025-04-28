import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binomtest
from .config import (
    K_RATIO, EPSILON, DIST_MEAN_THRESH, DIST_MAX_THRESH,
    SIGNIFICANCE_LEVEL, TP_PIPS, SL_PIPS, load_selected_features
)

# ===== フォールバック =====
_DEFAULT_FEATS = [
    'rsi_trend', 'adx', 'atr_change', 'aroon_down', 'aroon_up',
    'obv', 'price_pos', 'range', 'vol_roc5', 'vol_roc10'
]

def _get_feat_cols():
    cols = load_selected_features()
    return cols if cols else _DEFAULT_FEATS

def estimate_signals(train_df: pd.DataFrame, latest_row: pd.Series) -> tuple:
    feat_cols = _get_feat_cols()

    mask_all = train_df[feat_cols].notna().all(axis=1)

    # 最新行に欠損がある場合はスキップ
    if not latest_row[feat_cols].notna().all():
        return False, False, {}

    mb_tr = mask_all & train_df["label_buy"].notna()
    ms_tr = mask_all & train_df["label_sell"].notna()

    Xb_tr, yb_tr = train_df.loc[mb_tr, feat_cols].values, train_df.loc[mb_tr, "label_buy"].to_numpy()
    Xs_tr, ys_tr = train_df.loc[ms_tr, feat_cols].values, train_df.loc[ms_tr, "label_sell"].to_numpy()
    if len(Xb_tr) < 1 or len(Xs_tr) < 1:
        return False, False, {}

    scaler = StandardScaler()
    scaler.fit(train_df.loc[mb_tr | ms_tr, feat_cols].values)

    Xb_sc = scaler.transform(Xb_tr)
    Xs_sc = scaler.transform(Xs_tr)
    cur_sc = scaler.transform([[latest_row[c] for c in feat_cols]])

    # --- 動的 k 選定 (BUY) -----------------------------------------
    max_k_buy = max(1, int(len(Xb_tr) * K_RATIO))
    nbr_b = NearestNeighbors(n_neighbors=max_k_buy, n_jobs=-1).fit(Xb_sc)
    dist_all_b, idx_all_b = nbr_b.kneighbors(cur_sc, return_distance=True)
    dist_all_b = dist_all_b[0]
    idx_all_b  = idx_all_b[0]
    valid_k_b = [k for k in range(1, max_k_buy+1)
                 if dist_all_b[:k].mean() <= DIST_MEAN_THRESH and dist_all_b[:k].max() <= DIST_MAX_THRESH]

    # --- 動的 k 選定 (SELL) ----------------------------------------
    max_k_sell = max(1, int(len(Xs_tr) * K_RATIO))
    nbr_s = NearestNeighbors(n_neighbors=max_k_sell, n_jobs=-1).fit(Xs_sc)
    dist_all_s, idx_all_s = nbr_s.kneighbors(cur_sc, return_distance=True)
    dist_all_s = dist_all_s[0]
    idx_all_s  = idx_all_s[0]
    valid_k_s = [k for k in range(1, max_k_sell+1)
                 if dist_all_s[:k].mean() <= DIST_MEAN_THRESH and dist_all_s[:k].max() <= DIST_MAX_THRESH]

    if not valid_k_b and not valid_k_s:
        return False, False, {}

    k_nb_buy  = valid_k_b[-1] if valid_k_b else max_k_buy
    k_nb_sell = valid_k_s[-1] if valid_k_s else max_k_sell

    # BUY
    db = dist_all_b[:k_nb_buy]
    ib = idx_all_b[:k_nb_buy]
    buy_rate = ((1.0/(db+EPSILON)) * yb_tr[ib]).sum() / (1.0/(db+EPSILON)).sum() * 100
    mean_db, max_db = db.mean(), db.max()

    # SELL
    ds = dist_all_s[:k_nb_sell]
    is_ = idx_all_s[:k_nb_sell]
    sell_rate = ((1.0/(ds+EPSILON)) * ys_tr[is_]).sum() / (1.0/(ds+EPSILON)).sum() * 100
    mean_ds, max_ds = ds.mean(), ds.max()

    threshold = SL_PIPS / (TP_PIPS + SL_PIPS) * 100

    wins_b = int(yb_tr[ib].sum())
    wins_s = int(ys_tr[is_].sum())
    res_b  = binomtest(wins_b, n=k_nb_buy,  p=threshold/100, alternative='greater')
    res_s  = binomtest(wins_s, n=k_nb_sell, p=threshold/100, alternative='greater')
    pval_b, pval_s = res_b.pvalue, res_s.pvalue
    sig_b, sig_s   = (pval_b < SIGNIFICANCE_LEVEL), (pval_s < SIGNIFICANCE_LEVEL)

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

    buy_ok  = (buy_rate >= threshold and mean_db <= DIST_MEAN_THRESH and max_db <= DIST_MAX_THRESH and sig_b)
    sell_ok = (sell_rate >= threshold and mean_ds <= DIST_MEAN_THRESH and max_ds <= DIST_MAX_THRESH and sig_s)

    return buy_ok, sell_ok, metrics


def explain_reason(ok_flag, label, metrics):
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
