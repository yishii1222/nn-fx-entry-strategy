import numpy as np
from collections import Counter

def analyze_trades(trades, results, entry_times, holding_times, sim_start, end_dt):
    num_trades   = len(trades)
    wins         = sum(1 for p in trades if p > 0)
    losses       = sum(1 for p in trades if p < 0)
    win_rate     = wins / num_trades * 100 if num_trades > 0 else 0
    gross_profit = sum(p for p in trades if p > 0)
    gross_loss   = -sum(p for p in trades if p < 0)
    net_profit   = gross_profit - gross_loss
    pf           = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_return   = net_profit / num_trades if num_trades > 0 else 0

    days = (end_dt - sim_start).days + 1

    # 待機時間
    wait_times = []
    for i in range(1, len(entry_times)):
        diff = (entry_times[i] - entry_times[i-1]).total_seconds() / 60
        wait_times.append(diff)
    avg_wait   = sum(wait_times) / len(wait_times) if wait_times else 0
    median_wait= np.median(wait_times) if wait_times else 0
    std_wait   = np.std(wait_times) if wait_times else 0

    # 保有時間
    median_hold    = np.median(holding_times) if holding_times else 0
    std_hold       = np.std(holding_times) if holding_times else 0
    hold_times_win = [holding_times[i] for i,p in enumerate(trades) if p>0]
    hold_times_loss= [holding_times[i] for i,p in enumerate(trades) if p<0]
    avg_hold_win   = sum(hold_times_win)/len(hold_times_win) if hold_times_win else 0
    avg_hold_loss  = sum(hold_times_loss)/len(hold_times_loss) if hold_times_loss else 0

    # 連勝・連敗ストリーク
    win_streaks, loss_streaks = [], []
    cw, cl = 0, 0
    for p in trades:
        if p > 0:
            cw += 1
            if cl > 0:
                loss_streaks.append(cl); cl = 0
        elif p < 0:
            cl += 1
            if cw > 0:
                win_streaks.append(cw); cw = 0
    if cw>0: win_streaks.append(cw)
    if cl>0: loss_streaks.append(cl)
    win_streak_counts  = Counter(win_streaks)
    loss_streak_counts = Counter(loss_streaks)

    # シグナル連続ストリーク
    signal_history      = [r['signal'] for r in results]
    buy_signal_streaks  = []
    sell_signal_streaks = []
    current, count      = None, 0
    for s in signal_history:
        if s in ['BUY','SELL']:
            if s == current:
                count += 1
            else:
                if current == 'BUY':
                    buy_signal_streaks.append(count)
                elif current == 'SELL':
                    sell_signal_streaks.append(count)
                current, count = s, 1
        else:
            if current == 'BUY':
                buy_signal_streaks.append(count)
            elif current == 'SELL':
                sell_signal_streaks.append(count)
            current, count = None, 0
    if current == 'BUY':
        buy_signal_streaks.append(count)
    elif current == 'SELL':
        sell_signal_streaks.append(count)
    buy_signal_dist  = Counter(buy_signal_streaks)
    sell_signal_dist = Counter(sell_signal_streaks)

    # 買い／売り連敗ストリーク
    buy_loss_streaks, sell_loss_streaks = [], []
    bl, sl = 0, 0
    for r in results:
        if r['signal'] == 'BUY' and r['profit'] is not None:
            if r['profit'] < 0:
                bl += 1
            else:
                if bl > 0:
                    buy_loss_streaks.append(bl); bl = 0
        elif r['signal'] == 'SELL' and r['profit'] is not None:
            if r['profit'] < 0:
                sl += 1
            else:
                if sl > 0:
                    sell_loss_streaks.append(sl); sl = 0
    if bl > 0:
        buy_loss_streaks.append(bl)
    if sl > 0:
        sell_loss_streaks.append(sl)
    buy_loss_dist  = Counter(buy_loss_streaks)
    sell_loss_dist = Counter(sell_loss_streaks)

    return {
        'num_trades': num_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'net_profit': net_profit,
        'pf': pf,
        'avg_return': avg_return,
        'days': days,
        'avg_wait': avg_wait,
        'median_wait': median_wait,
        'std_wait': std_wait,
        'median_hold': median_hold,
        'std_hold': std_hold,
        'avg_hold_win': avg_hold_win,
        'avg_hold_loss': avg_hold_loss,
        'win_streak_counts': win_streak_counts,
        'loss_streak_counts': loss_streak_counts,
        'buy_signal_dist': buy_signal_dist,
        'sell_signal_dist': sell_signal_dist,
        'buy_loss_dist': buy_loss_dist,
        'sell_loss_dist': sell_loss_dist,
    }