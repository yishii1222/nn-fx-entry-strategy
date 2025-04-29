import pandas as pd
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
            elapsed = now_loop - start_time
            avg_time = elapsed / processed
            remaining = total - processed
            eta = now_loop + avg_time * remaining
            print(
                f"\rProgress: {processed}/{total} ({processed/total*100:.1f}%), "
                f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}, Elapsed: {str(elapsed).split('.')[0]}",
                end="", flush=True
            )

        # 学習ウィンドウ
        train_start = current_time - BDay(DAYS_BACK)
        train_end = current_time - timedelta(minutes=1)
        train_df = df_all.loc[train_start:train_end]

        # ======== データリーク防止 ==================================
        if not train_df.empty:
            elapsed_min = (current_time - train_df.index).total_seconds() / 60
            elapsed_min = pd.Series(elapsed_min, index=train_df.index)
            # まだ「未来」に相当するラベルは無効化
            leak_b = train_df['time_buy']  > elapsed_min
            leak_s = train_df['time_sell'] > elapsed_min
            train_df.loc[leak_b, 'label_buy']  = pd.NA
            train_df.loc[leak_s, 'label_sell'] = pd.NA
        # ==========================================================

        if len(train_df) < 50:
            results.append({'time': current_time, 'signal': 'NONE', 'profit': None})
            continue

        # シグナル判定
        buy_ok, sell_ok, _ = estimate_signals(train_df, df_all.loc[current_time])
        # ラベル未付与ならシグナル無効化（profit=None 率低減）
        if buy_ok and pd.isna(df_all.at[current_time, 'label_buy']):
            buy_ok = False
        if sell_ok and pd.isna(df_all.at[current_time, 'label_sell']):
            sell_ok = False

        # ポジション保有中はシグナル自体を無効化
        if skip_until and current_time <= skip_until:
            results.append({'time': current_time, 'signal': 'NONE', 'profit': None})
            continue

        # BUY エントリー
        if buy_ok:
            label = df_all.at[current_time, 'label_buy']
            time_offset = df_all.at[current_time, 'time_buy']
            entry_times.append(current_time)
            holding_times.append(time_offset if not pd.isna(time_offset) else 0)
            raw_profit = TP_PIPS if label == 1 else -SL_PIPS
            adj_profit = raw_profit - SPREAD_PIPS
            trades.append(adj_profit)
            results.append({'time': current_time, 'signal': 'BUY', 'profit': adj_profit})
            if not pd.isna(time_offset):
                skip_until = current_time + timedelta(minutes=int(time_offset))

        # SELL エントリー
        elif sell_ok:
            label = df_all.at[current_time, 'label_sell']
            time_offset = df_all.at[current_time, 'time_sell']
            entry_times.append(current_time)
            holding_times.append(time_offset if not pd.isna(time_offset) else 0)
            raw_profit = TP_PIPS if label == 1 else -SL_PIPS
            adj_profit = raw_profit - SPREAD_PIPS
            trades.append(adj_profit)
            results.append({'time': current_time, 'signal': 'SELL', 'profit': adj_profit})
            if not pd.isna(time_offset):
                skip_until = current_time + timedelta(minutes=int(time_offset))

        # ノーエントリー
        else:
            results.append({'time': current_time, 'signal': 'NONE', 'profit': None})

    print()  # Progress 改行
    return trades, results, entry_times, holding_times
