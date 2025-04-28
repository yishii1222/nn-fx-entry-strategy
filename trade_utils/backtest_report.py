import os
import matplotlib.pyplot as plt
import numpy as np

def generate_backtest_report(metrics, trades):
    print("=== バックテスト結果 ===")
    print(f"テスト期間          : {metrics['start_date']} ~ {metrics['end_date']}")
    print(f"テスト日数          : {metrics['days']}")
    print(f"トレード回数        : {metrics['num_trades']}")
    print(f"勝率               : {metrics['win_rate']:.2f}% ({metrics['wins']}/{metrics['num_trades']})")
    print(f"純利益（pips）      : {metrics['net_profit']}")
    print(f"日平均純利益（pips） : {metrics['net_profit']/metrics['days']:.0f}")
    print(f"利益ファクター (PF) : {metrics['pf']:.2f}")
    print(f"平均リターン (pips) : {metrics['avg_return']:.2f}")
    cum = np.cumsum(trades)
    drawdown = np.maximum.accumulate(cum) - cum
    print(f"最大ドローダウン     : {drawdown.max():.2f}")
    print(f"平均待機分数        : {metrics['avg_wait']:.2f}")
    print(f"中央値待機分数      : {metrics['median_wait']:.2f}")
    print(f"待機時間標準偏差    : {metrics['std_wait']:.2f}")
    print(f"保有分数中央値      : {metrics['median_hold']:.2f}")
    print(f"保有時間標準偏差    : {metrics['std_hold']:.2f}")
    print(f"平均保有分数 (勝利)   : {metrics['avg_hold_win']:.2f}")
    print(f"平均保有分数 (敗北)   : {metrics['avg_hold_loss']:.2f}")
    print(f"連勝ストリーク(分布): {metrics['win_streak_counts']}")
    print(f"連敗ストリーク(分布): {metrics['loss_streak_counts']}")
    print(f"シグナルBUY(分布)  : {metrics['buy_signal_dist']}")
    print(f"シグナルSELL(分布) : {metrics['sell_signal_dist']}")
    print(f"買い連敗ストリーク(分布): {metrics['buy_loss_dist']}")
    print(f"売り連敗ストリーク(分布): {metrics['sell_loss_dist']}")
    print(f"決着なしトレード数    : {metrics.get('no_decision_count', 0)}")
    print(f"決着なし割合         : {metrics.get('no_decision_rate', 0):.2f}%")

    # 成長曲線グラフ保存
    plt.figure()
    plt.plot(range(1, len(cum)+1), cum)
    plt.title("pips")
    plt.xlabel("Trade No")
    plt.ylabel("pips")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("report", exist_ok=True)
    plt.savefig(os.path.join("report", "growth_curve.png"))
    plt.close()