import time
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import datetime, timezone, timedelta
from trade_utils.config import ACCESS_TOKEN, INSTRUMENT, DAYS_BACK, JST
from trade_utils.data_fetch import fetch_1min_data
from trade_utils.features import compute_features_and_labels
from trade_utils.signals import estimate_signals, explain_reason
from trade_utils.notifier import send_notification, beep

# ====== 実行モード設定 ======
LOOP_MODE = True


def execute_analysis():
    now = datetime.now(timezone.utc)
    start = pd.to_datetime(now) - BDay(DAYS_BACK)
    df = fetch_1min_data(start, pd.to_datetime(now), ACCESS_TOKEN, INSTRUMENT)

    if df.empty or len(df) < 50:
        print("データ取得失敗または不足")
        return

    df = compute_features_and_labels(df)
    runtime_feats = [
        c for c in df.columns
        if c not in [
            "open", "high", "low", "close", "volume",
            "label_buy", "label_sell", "time_buy", "time_sell"
        ]
    ]

    buy_ok, sell_ok, metrics = estimate_signals(df, df.iloc[-1])

    now_exec = datetime.now(timezone.utc).astimezone(JST)
    lt_time = df.index[-1].astimezone(JST)
    print("=" * 30 + " 現在の局面分析 " + "=" * 30)
    print("分析足確定時刻 :", lt_time.strftime("%Y-%m-%d %H:%M:%S"), "JST")
    print("実行完了時刻   ：", now_exec.strftime("%Y-%m-%d %H:%M:%S"), "JST")
    print("-" * 60)
    print("使用特徴量        :", ", ".join(sorted(runtime_feats)))

    if not metrics:
        print("有効サンプル不足のためシグナル算出不可")
        return

    print(f"損益分岐勝率            : {metrics['threshold']:.2f}%")
    print(f"モデル推定 Buy勝率      : {metrics['buy_rate']:.2f}%  平均距離{metrics['mean_db']:.3f}, 最大距離{metrics['max_db']:.3f}, n数{metrics['n_k_nb_buy']:.0f}")
    print(f"モデル推定 Sell勝率     : {metrics['sell_rate']:.2f}%  平均距離{metrics['mean_ds']:.3f}, 最大距離{metrics['max_ds']:.3f}, n数{metrics['n_k_nb_sell']:.0f}")
    print(f"Buy p-value            : {metrics['pval_buy']:.4f}")
    print(f"Sell p-value           : {metrics['pval_sell']:.4f}")
    print(f"判定足終値     : {df.iloc[-1]['close']:.3f}")

    if buy_ok and not sell_ok:
        print("→ 戦略：BUY 推奨")
        send_notification("FX シグナル", "BUY 推奨")
        beep()
    elif sell_ok and not buy_ok:
        print("→ 戦略：SELL 推奨")
        send_notification("FX シグナル", "SELL 推奨")
        beep()
    elif buy_ok and sell_ok:
        strategy = "BUY 推奨" if metrics['buy_rate'] > metrics['sell_rate'] else "SELL 推奨"
        print(f"→ 戦略：{strategy}")
        send_notification("FX シグナル", strategy)
        beep()
    else:
        print("→ 戦略：様子見")

    explain_reason(buy_ok, "buy", metrics)
    explain_reason(sell_ok, "sell", metrics)


if LOOP_MODE:
    while True:
        loop_start = datetime.now(timezone.utc)
        execute_analysis()
        next_run = (loop_start + timedelta(minutes=1)).replace(second=0, microsecond=0)
        sleep_sec = (next_run - datetime.now(timezone.utc)).total_seconds()
        if sleep_sec > 0:
            time.sleep(sleep_sec)
else:
    execute_analysis()
