import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from trade_utils.config import TP_PIPS, SL_PIPS, SPREAD_PIPS, DAYS_BACK
from trade_utils.signals import estimate_signals

def simulate_trades(df_all: pd.DataFrame, sim_start, end_dt):
    trades = []
    results = []
    entry_times = []
    holding_times = []
    valid_times = [t for t in df_all.index if sim_start <= t <= end_dt]
    total = len(valid_times)
    processed = 0
    interval = 500
    start_time = datetime.now()
    skip_until = None

    for current_time in valid_times:
        processed += 1
        if processed % interval == 0 or processed == total:
            now_loop = datetime.now()
            elapsed  = now_loop - start_time
            avg_time = elapsed / processed
            remaining= total - processed
            eta      = now_loop + avg_time * remaining
            print(f"\rProgress: {processed}/{total} ({processed/total*100:.1f}%), ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}, Elapsed: {str(elapsed).split('.')[0]}", end="", flush=True)

        # 学習ウィンドウ
        train_start = current_time - BDay(DAYS_BACK)
        train_end   = current_time - timedelta(minutes=1)
        train_df    = df_all.loc[train_start:train_end]
        if len(train_df) < 50:
            results.append({'time': current_time, 'signal': 'NONE', 'profit': None})
            continue

        buy_ok, sell_ok, _ = estimate_signals(train_df, df_all.loc[current_time])

        # ポジション保有中はエントリーのみスキップ
        if skip_until and current_time <= skip_until:
            sig = 'BUY' if buy_ok else 'SELL' if sell_ok else 'NONE'
            results.append({'time': current_time, 'signal': sig, 'profit': None})
            continue

        # BUY シグナル処理
        if buy_ok:
            label       = df_all.at[current_time, 'label_buy']
            time_offset = df_all.at[current_time, 'time_buy']
            if not pd.isna(label):
                entry_times.append(current_time)
                holding_times.append(time_offset if not pd.isna(time_offset) else 0)
                raw_profit = TP_PIPS if label == 1 else -SL_PIPS
                adj_profit = raw_profit - SPREAD_PIPS
                trades.append(adj_profit)
                results.append({'time': current_time, 'signal': 'BUY',  'profit': adj_profit})
                if not pd.isna(time_offset):
                    skip_until = current_time + timedelta(minutes=int(time_offset))
            else:
                results.append({'time': current_time, 'signal': 'BUY', 'profit': None})

        # SELL シグナル処理
        elif sell_ok:
            label       = df_all.at[current_time, 'label_sell']
            time_offset = df_all.at[current_time, 'time_sell']
            if not pd.isna(label):
                entry_times.append(current_time)
                holding_times.append(time_offset if not pd.isna(time_offset) else 0)
                raw_profit = TP_PIPS if label == 1 else -SL_PIPS
                adj_profit = raw_profit - SPREAD_PIPS
                trades.append(adj_profit)
                results.append({'time': current_time, 'signal': 'SELL', 'profit': adj_profit})
                if not pd.isna(time_offset):
                    skip_until = current_time + timedelta(minutes=int(time_offset))
            else:
                results.append({'time': current_time, 'signal': 'SELL', 'profit': None})

        # ノーシグナル
        else:
            results.append({'time': current_time, 'signal': 'NONE', 'profit': None})

    print()
    return trades, results, entry_times, holding_times