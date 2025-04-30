# test_fetch.py

import sys
from datetime import datetime, timezone
import pandas as pd
from pandas.tseries.offsets import BDay

#――――――――――――――――――――――――――――
# バックテスト側の DAYS_BACK を短縮（テスト用）
#――――――――――――――――――――――――――――
import trade_utils.config as cfg
cfg.DAYS_BACK = 1

from trade_utils.config import ACCESS_TOKEN, INSTRUMENT, DAYS_BACK, JST
from trade_utils.data_fetch import fetch_1min_data
from trade_utils.backtest_data import load_backtest_data

def check_df(df: pd.DataFrame, label: str):
    print(f"\n--- {label} ---")
    print(f"Rows:       {len(df)}")
    # 重複タイムスタンプ
    dup = df.index.duplicated().sum()
    print(f"Duplicates: {dup}")
    # 1分以上のギャップ
    diffs = df.index.to_series().diff().dropna()
    gaps  = diffs[diffs > pd.Timedelta(minutes=1)]
    print(f"Gaps:       {len(gaps)}")
    if not gaps.empty:
        print("Gap samples:", gaps.head().to_list())

def main():
    #――― Live fetch test ―――
    now       = datetime.now(timezone.utc)
    start_live= pd.to_datetime(now) - BDay(DAYS_BACK)
    df_live   = fetch_1min_data(start_live, pd.to_datetime(now), ACCESS_TOKEN, INSTRUMENT)
    check_df(df_live, "Live data fetch")

    #――― Backtest fetch test ―――
    # backtest.py と同じロジックで load_backtest_data を呼び出し
    # start_date, end_date は「今」だけを見るように同一日に設定
    now_jst   = now.astimezone(JST)
    start_date= now_jst.strftime("%Y-%m-%dT00:00:00")
    end_date  = now_jst.strftime("%Y-%m-%dT%H:%M:%S")

    df_bt, sim_start, end_dt = load_backtest_data(start_date, end_date)
    if df_bt.empty:
        print("\nBacktest data fetch: No data returned")
        sys.exit(1)
    check_df(df_bt, "Backtest data fetch")

if __name__ == "__main__":
    main()
