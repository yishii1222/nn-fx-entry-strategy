import os
import json
from datetime import timedelta, timezone

# ====== 可変設定値 ======
TP_PIPS             = 10               # Before : 5
SL_PIPS             = 10               # Before : 3
DAYS_BACK           = 10               # Before : 5
K_RATIO             = 0.10             # Before : 0.01
DIST_MEAN_THRESH    = 1.0
DIST_MAX_THRESH     = 2.0
EPSILON             = 1e-6
SIGNIFICANCE_LEVEL  = 0.05

# ====== 固定設定値 ======
SPREAD_PIPS         = 0.2
ACCESS_TOKEN        = '0277ce3e38b48230acfb6ed493c43a70-c4952ef6136325c9f5c713d1d218fdaa'
INSTRUMENT          = 'USD_JPY'
JST                 = timezone(timedelta(hours=9))
THRESH_PATH         = "optuna_config.json"

# Optuna時に設定値取得
def load_thresholds_from_file():
    global DIST_MEAN_THRESH, DIST_MAX_THRESH
    if os.path.exists(THRESH_PATH):
        with open(THRESH_PATH, "r") as f:
            config = json.load(f)
            DIST_MEAN_THRESH = config.get("mean_thresh", DIST_MEAN_THRESH)
            DIST_MAX_THRESH  = config.get("max_thresh", DIST_MAX_THRESH)