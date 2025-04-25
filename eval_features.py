import os
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import datetime, timezone
from trade_utils.config import ACCESS_TOKEN, INSTRUMENT, DAYS_BACK
from trade_utils.data_fetch import fetch_1min_data
from trade_utils.features import compute_features_and_labels
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 閾値設定
CORR_THRESHOLD = 0.8
VIF_THRESHOLD  = 10.0


def calculate_vif(df: pd.DataFrame) -> pd.Series:
    """
    DataFrame の各特徴量について VIF (分散膨張係数) を計算する
    """
    vif_values = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return pd.Series(vif_values, index=df.columns)


def main():
    # 出力ディレクトリ
    out_dir = "eval_features"
    os.makedirs(out_dir, exist_ok=True)

    # データ取得
    now   = pd.to_datetime(datetime.now(timezone.utc))
    start = now - BDay(DAYS_BACK)
    df    = fetch_1min_data(start, now, ACCESS_TOKEN, INSTRUMENT)
    if df.empty:
        print("データ取得失敗または不足")
        return

    # 特徴量計算
    df = compute_features_and_labels(df)

    # 評価対象特徴量
    # eval_features.py の feat_cols に指定する全特徴量一覧
    feat_cols = [
        "rsi",
        "rsi_trend",
        "adx",
        "atr",
        "atr_change",
        "ret1",
        "ret5",
        "ret15",
        "roc3",
        "roc10",
        "sma5",
        "sma10",
        "sma20",
        "sma40",
        "sma100",
        "ema5",
        "ema10",
        "ema20",
        "ema40",
        "ema100",
        "sma5_slope",
        "sma10_slope",
        "sma20_slope",
        "sma40_slope",
        "sma100_slope",
        "ema5_slope",
        "ema10_slope",
        "ema20_slope",
        "ema40_slope",
        "ema100_slope",
        "dist_to_sma5",
        "dist_to_sma10",
        "dist_to_sma20",
        "dist_to_sma40",
        "dist_to_sma100",
        "dist_to_ema5",
        "dist_to_ema10",
        "dist_to_ema20",
        "dist_to_ema40",
        "dist_to_ema100",
        "bb_mid",
        "bb_std",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_percent",
        "ema12",
        "ema26",
        "macd",
        "macd_signal",
        "macd_hist",
        "sto_k",
        "sto_d",
        "cci",
        "obv",
        "aroon_up",
        "aroon_down",
        "willr",
        "price_pos",
        "range",
        "atr_pct",
        "vol_roc5",
        "vol_roc10",
    ]

    feat_df = df[feat_cols].dropna()

    # 相関行列
    corr = feat_df.corr()
    corr.to_csv(os.path.join(out_dir, "feature_correlation.csv"))

    # VIF 計算
    vif = calculate_vif(feat_df)
    vif.to_csv(os.path.join(out_dir, "feature_vif.csv"), header=False)

    # 多重共線性に基づく除外候補抽出
    drop_set = set()
    # 相関が高いペアから VIF の大きい方を除外
    for i, f1 in enumerate(feat_cols):
        for f2 in feat_cols[i+1:]:
            if abs(corr.loc[f1, f2]) > CORR_THRESHOLD:
                # VIF が大きい方を除外
                if vif[f1] >= vif[f2]:
                    drop_set.add(f1)
                else:
                    drop_set.add(f2)
    # VIF が閾値超の特徴量を除外
    for f in feat_cols:
        if vif[f] > VIF_THRESHOLD:
            drop_set.add(f)

    # 最終選択特徴量
    selected = [f for f in feat_cols if f not in drop_set]
    # 結果出力
    print("=== 除外された特徴量 ===")
    print(sorted(drop_set))
    print("=== 選択された特徴量 ===")
    print(sorted(selected))

    # 選択リストを保存
    pd.Series(selected).to_csv(os.path.join(out_dir, "selected_features.csv"), index=False, header=False)

if __name__ == "__main__":
    main()
