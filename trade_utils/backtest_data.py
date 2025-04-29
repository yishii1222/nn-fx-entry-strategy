import pandas as pd
from datetime import timezone as _timezone
from pandas.tseries.offsets import BDay
from trade_utils.config import ACCESS_TOKEN, INSTRUMENT, DAYS_BACK, JST
from trade_utils.data_fetch import fetch_1min_data
from trade_utils.features import compute_features_and_labels

def load_backtest_data(start_date: str, end_date: str) -> tuple:

    # 日本時間(JST)をUTCに変換
    end_dt    = pd.to_datetime(end_date).tz_localize(JST).tz_convert(_timezone.utc)
    sim_start = pd.to_datetime(start_date).tz_localize(JST).tz_convert(_timezone.utc)
    # 実営業日ベースで過去 DAYS_BACK 日分を取得
    fetch_start = sim_start - BDay(DAYS_BACK)

    # データ取得
    df_all = fetch_1min_data(fetch_start, end_dt, ACCESS_TOKEN, INSTRUMENT)
    if df_all.empty:
        print("データ取得失敗")
        return pd.DataFrame(), None, None

    # 特徴量計算＋ラベル付け
    df_all = compute_features_and_labels(df_all)
    return df_all, sim_start, end_dt