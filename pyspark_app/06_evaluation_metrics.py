# 06_evaluation_metrics.py
# Goal: Load the latest trained model + its feature set, re-split data with the same seed,
# evaluate on the test split, compute common metrics + confusion matrix, and log to CSV.

import csv
import os
import time
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel

# I use a fixed split/seed so results are consistent with training runs 
SEED = 42
SPLIT = [0.8, 0.2]

# Paths for reading the training metadata and writing eval outputs
RUNS_DIR = "../runs"
TRAINING_REPORT = os.path.join(RUNS_DIR, "training_report.csv")
EVAL_REPORT = os.path.join(RUNS_DIR, "evaluation_report.csv")

# Helper: grab the last row from training_report.csv (latest experiment meta) 
def read_last_training_row(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f)]
    if len(rows) < 2:
        raise RuntimeError("training_report.csv has no data rows.")
    header = rows[0]
    last = rows[-1]
    meta = dict(zip(header, last))
    return meta  # contains: run_id, features_path, model_path, rows_test, ...

# Pull the latest training metadata (so I evaluate the same artifacts)
meta = read_last_training_row(TRAINING_REPORT)
features_path = meta["features_path"]
model_path    = meta["model_path"]
run_id_train  = meta["run_id"]

# I measure wall-clock time for this evaluation step
t0 = time.time()

# Spin up Spark (same shuffle partitions I used elsewhere)
spark = (
    SparkSession.builder
    .appName("06_Evaluation_Metrics")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)

# Keep a few Spark env settings for the report (useful when comparing cores/memory)
spark_master     = spark.sparkContext.master
driver_memory    = spark.conf.get("spark.driver.memory",   "unknown")
executor_memory  = spark.conf.get("spark.executor.memory", "unknown")

print(f"Using features: {features_path}")
print(f"Loading model:  {model_path}")

# Load features and keep only what I need to score 
df = spark.read.parquet(features_path).select("label", "features")

# Recreate the same split (important: same SEED as training)
train, test = df.randomSplit(SPLIT, seed=SEED)
rows_test  = test.count()
rows_train = train.count()
rows_total = rows_train + rows_test
print(f"Rows total/train/test = {rows_total}/{rows_train}/{rows_test}")

# Load the trained Logistic Regression model and score the test set 
model = LogisticRegressionModel.load(model_path)
pred  = model.transform(test).select("label", "prediction", "probability")

# Standard metrics (overall)
acc_eval = MulticlassClassificationEvaluator(metricName="accuracy",           labelCol="label", predictionCol="prediction")
f1_eval  = MulticlassClassificationEvaluator(metricName="f1",                 labelCol="label", predictionCol="prediction")
wp_eval  = MulticlassClassificationEvaluator(metricName="weightedPrecision",  labelCol="label", predictionCol="prediction")
wr_eval  = MulticlassClassificationEvaluator(metricName="weightedRecall",     labelCol="label", predictionCol="prediction")

acc   = acc_eval.evaluate(pred)
f1    = f1_eval.evaluate(pred)
wprec = wp_eval.evaluate(pred)
wrec  = wr_eval.evaluate(pred)

# ROC-AUC (binary, using probabilities)
auc_eval = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label", metricName="areaUnderROC")
auc = auc_eval.evaluate(pred)

# --- Confusion matrix counts for positive=1.0 / negative=0.0 ---
tp = pred.filter((col("label") == 1.0) & (col("prediction") == 1.0)).count()
tn = pred.filter((col("label") == 0.0) & (col("prediction") == 0.0)).count()
fp = pred.filter((col("label") == 0.0) & (col("prediction") == 1.0)).count()
fn = pred.filter((col("label") == 1.0) & (col("prediction") == 0.0)).count()

# Class-1 (positive) precision/recall (nice to report explicitly)
pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
pos_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

# How long this evaluation took (seconds)
duration_sec = round(time.time() - t0, 2)

# Append a row to evaluation_report.csv
os.makedirs(RUNS_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
eval_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

out_cols = [
    "eval_run_id","start_ts","duration_sec","spark_master","driver_memory","executor_memory",
    "model_path","features_path","rows_total","rows_train","rows_test",
    "accuracy","f1","weighted_precision","weighted_recall","auc",
    "tp","fp","tn","fn","pos_precision","pos_recall","source_training_run_id"
]

row = [
    eval_run_id, ts, f"{duration_sec:.2f}", spark_master, driver_memory, executor_memory,
    model_path, features_path, rows_total, rows_train, rows_test,
    f"{acc:.6f}", f"{f1:.6f}", f"{wprec:.6f}", f"{wrec:.6f}", f"{auc:.6f}",
    tp, fp, tn, fn, f"{pos_precision:.6f}", f"{pos_recall:.6f}", run_id_train
]

write_header = not os.path.exists(EVAL_REPORT)
with open(EVAL_REPORT, "a", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(out_cols)
    w.writerow(row)

print(f"Evaluation written to: {EVAL_REPORT}")
spark.stop()
