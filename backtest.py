from trade_utils.backtest_data import load_backtest_data
from trade_utils.backtest_sim import simulate_trades
from trade_utils.backtest_analysis import analyze_trades
from trade_utils.backtest_report import generate_backtest_report
from trade_utils.config import DAYS_BACK, LABEL_MAX_MINUTES

import csv
import os
from datetime import datetime

# シミュレーション対象期間（日本時間 JST 指定）
START_DATE = '2025-04-21T00:00:00'
END_DATE   = '2025-04-25T23:59:59'

def save_backtest_log(start_date: str,
                      end_date: str,
                      days_back: int,
                      label_max_minutes: int,
                      features: list[str],
                      metrics: dict,
                      log_path: str = "backtest_log.csv") -> None:
    """
    バックテスト 1 回分の主要指標を CSV へ追記保存する。
    ファイルが無い場合はヘッダー行を自動生成する。
    """
    header = [
        "run_at", "start_date", "end_date", "days_back", "label_max_minutes",
        "features", "win_rate", "wins", "num_trades", "net_profit",
        "pf", "avg_return", "no_decision_rate"
    ]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        start_date,
        end_date,
        days_back,
        label_max_minutes,
        ";".join(sorted(features)),
        f"{metrics.get('win_rate', 0):.2f}",
        metrics.get("wins", 0),
        metrics.get("num_trades", 0),
        metrics.get("net_profit", 0),
        f"{metrics.get('pf', 0):.2f}" if metrics.get("pf") != float("inf") else "inf",
        f"{metrics.get('avg_return', 0):.2f}",
        f"{metrics.get('no_decision_rate', 0):.2f}"
    ]

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def main():
    df_all, sim_start, end_dt = load_backtest_data(START_DATE, END_DATE)
    if df_all.empty:
        return

    runtime_feats = [
        c for c in df_all.columns
        if c not in [
            "open", "high", "low", "close", "volume",
            "label_buy", "label_sell", "time_buy", "time_sell"
        ]
    ]
    print("使用特徴量        :", ", ".join(sorted(runtime_feats)))

    trades, results, entry_times, holding_times = simulate_trades(df_all, sim_start, end_dt)
    metrics = analyze_trades(trades, results, entry_times, holding_times, sim_start, end_dt)
    metrics['start_date'] = START_DATE
    metrics['end_date']   = END_DATE

    generate_backtest_report(metrics, trades)

    save_backtest_log(
        START_DATE,
        END_DATE,
        DAYS_BACK,
        LABEL_MAX_MINUTES,
        runtime_feats,
        metrics
    )


def run():
    main()


if __name__ == '__main__':
    run()
