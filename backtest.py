from trade_utils.backtest_data import load_backtest_data
from trade_utils.backtest_sim import simulate_trades
from trade_utils.backtest_analysis import analyze_trades
from trade_utils.backtest_report import generate_backtest_report

# シミュレーション対象期間（日本時間JST指定）
START_DATE   = '2025-04-21T00:00:00'
END_DATE     = '2025-04-24T23:59:59'

def main():
    df_all, sim_start, end_dt = load_backtest_data(START_DATE, END_DATE)
    if df_all.empty:
        return
    trades, results, entry_times, holding_times = simulate_trades(df_all, sim_start, end_dt)
    metrics = analyze_trades(trades, results, entry_times, holding_times, sim_start, end_dt)
    metrics['start_date'] = START_DATE
    metrics['end_date'] = END_DATE
    generate_backtest_report(metrics, trades)

def run():
    main()

if __name__ == '__main__':
    run()
