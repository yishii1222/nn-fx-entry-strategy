import os
import json
from datetime import timedelta, timezone

# ====== 可変設定値 ======
TP_PIPS             = 10
SL_PIPS             = 10
DAYS_BACK           = 20               # Before : 10
K_RATIO             = 0.10
DIST_MEAN_THRESH    = 1.0
DIST_MAX_THRESH     = 2.0
EPSILON             = 1e-6
SIGNIFICANCE_LEVEL  = 0.05
LABEL_MAX_MINUTES   = 90               # Before : 60 -> 90 ※ 要検討 未解決

# ====== 固定設定値 ======
SPREAD_PIPS         = 0.2
ACCESS_TOKEN        = '0277ce3e38b48230acfb6ed493c43a70-c4952ef6136325c9f5c713d1d218fdaa'
INSTRUMENT          = 'USD_JPY'
JST                 = timezone(timedelta(hours=9))
THRESH_PATH         = "optuna_config.json"
FEATURE_CONF_PATH   = "selected_features.json"

# Optuna時に設定値取得
def load_thresholds_from_file():
    global DIST_MEAN_THRESH, DIST_MAX_THRESH
    if os.path.exists(THRESH_PATH):
        with open(THRESH_PATH, "r") as f:
            config = json.load(f)
            DIST_MEAN_THRESH = config.get("mean_thresh", DIST_MEAN_THRESH)
            DIST_MAX_THRESH  = config.get("max_thresh", DIST_MAX_THRESH)

# 選択特徴量読み込み
def load_selected_features():
    """
    JSON から選択特徴量リストを読み込む。
    無い場合は空リストを返し、呼び出し側でフォールバック。
    """
    if os.path.exists(FEATURE_CONF_PATH):
        with open(FEATURE_CONF_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
