import os
import json
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import datetime, timezone
from trade_utils.config import ACCESS_TOKEN, INSTRUMENT, DAYS_BACK, FEATURE_CONF_PATH
from trade_utils.features import compute_features_and_labels, list_available_features
from statsmodels.stats.outliers_influence import variance_inflation_factor
from trade_utils.data_fetch import fetch_1min_data
from scipy.stats import pointbiserialr

# ====== 閾値設定 ======
CORR_THRESHOLD = 0.8
VIF_THRESHOLD  = 10.0
PVAL_THRESHOLD = 0.10  # ターゲット相関検定の p 値閾値

# ====== ホールドアウト設定 ======
HOLDOUT_RANGE_DAYS = 60   # 特徴選択用期間（営業日数）
MAX_BACKTEST_DAYS  = 22   # バックテストが参照しうる最大期間（営業日数）


def calculate_vif(df: pd.DataFrame) -> pd.Series:
    """
    DataFrame の各特徴量について VIF (分散膨張係数) を計算する
    """
    vif_values = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return pd.Series(vif_values, index=df.columns)


def main():
    out_dir = "eval_features"
    os.makedirs(out_dir, exist_ok=True)

    # データ取得範囲設定
    now = pd.to_datetime(datetime.now(timezone.utc))
    holdout_end   = now - BDay(DAYS_BACK + MAX_BACKTEST_DAYS)
    holdout_start = holdout_end - BDay(HOLDOUT_RANGE_DAYS)
    df = fetch_1min_data(holdout_start, holdout_end, ACCESS_TOKEN, INSTRUMENT)
    if df.empty:
        print("データ取得失敗または不足")
        return

    # === 変更点: 全候補特徴量を自動取得 ===========================
    feat_cols = list_available_features()
    df = compute_features_and_labels(df, selected_features=feat_cols)

    # 欠損を除外
    feat_df = df[feat_cols].dropna()

    # 相関行列出力
    corr = feat_df.corr()
    corr.to_csv(os.path.join(out_dir, "feature_correlation.csv"))

    # VIF 計算
    vif = calculate_vif(feat_df)
    vif.to_csv(os.path.join(out_dir, "feature_vif.csv"), header=False)

    # 除外候補セット生成
    drop_set = set()
    for i, f1 in enumerate(feat_cols):
        for f2 in feat_cols[i + 1:]:
            if abs(corr.loc[f1, f2]) > CORR_THRESHOLD:
                drop_set.add(f1 if vif[f1] >= vif[f2] else f2)
    for f in feat_cols:
        if vif[f] > VIF_THRESHOLD:
            drop_set.add(f)

    # ターゲット相関検定
    df_target = feat_df.copy()
    df_target['label'] = df['label_buy'].reindex(feat_df.index)
    df_target = df_target.dropna(subset=['label'])
    target_drop = set()
    for f in feat_cols:
        r, p = pointbiserialr(df_target[f], df_target['label'])
        if p >= PVAL_THRESHOLD:
            target_drop.add(f)
    drop_set |= target_drop

    # 結果表示
    print("=== 除外された特徴量 ===")
    print(sorted(drop_set))
    print("=== ターゲット相関検定で除外された特徴量 ===")
    print(sorted(target_drop))
    selected = [f for f in feat_cols if f not in drop_set]
    print("=== 選択された特徴量 ===")
    print(sorted(selected))

    # 選択リスト保存
    with open(FEATURE_CONF_PATH, "w", encoding="utf-8") as f_json:
        json.dump(selected, f_json, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
