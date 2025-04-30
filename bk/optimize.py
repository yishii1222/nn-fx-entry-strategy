# ※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※
# テストトライのため、以下は探索を2つのみで実施する処理
# ※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※※
import json
import optuna
import importlib.util
from io import StringIO
import sys
import os

os.environ["OPTUNA_RUN"] = "1"

def objective(trial):
    mean_thresh = trial.suggest_float("mean_thresh", 0.5, 2.0, step=0.1)
    max_thresh = trial.suggest_float("max_thresh", 1.0, 3.0, step=0.1)

    # 探索設定をJSONファイルに保存
    with open("optuna_config.json", "w") as f:
        json.dump({"mean_thresh": mean_thresh, "max_thresh": max_thresh}, f)

    # backtest.py をimportして run() 実行
    spec = importlib.util.spec_from_file_location("backtest", "../backtest.py")
    backtest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backtest)

    temp_stdout = StringIO()
    sys.stdout = temp_stdout

    backtest.run()

    sys.stdout = sys.__stdout__
    output = temp_stdout.getvalue()
    print(output)

    for line in output.splitlines():
        if "利益ファクター" in line:
            try:
                pf = float(line.split(":")[1].strip())
                return pf
            except (IndexError, ValueError) as e:
                print(f"[ParseError] {e}")
                return -1.0
    return -1.0

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("==== ベストパラメータ ====")
print(study.best_params)
