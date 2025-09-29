# 05_model_training.py
# Goal: Train a Logistic Regression model on the features (from step 04).
# Steps: load latest features parquet -> split train/test -> train model -> evaluate -> save results.

import os, csv, time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel

# Directories for saving runs, reports, and artifacts
RUNS_DIR = "../runs"
REPORT_TRAIN = os.path.join(RUNS_DIR, "training_report.csv")
ARTIFACTS_DIR = "../artifacts"

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Helper: get the latest features parquet path
def get_latest_features_path():
    path_csv = os.path.join(RUNS_DIR, "feature_report.csv")
    if not os.path.exists(path_csv):
        raise FileNotFoundError("runs/feature_report.csv was not found. Run the Feature Engineering step first.")
    last = None
    with open(path_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:  # just keep the last row (latest run)
            last = row
    if not last or "out_parquet" not in last:
        raise RuntimeError("Could not read out_parquet from feature_report.csv")
    return last["out_parquet"]

# Start Spark session
spark = (
    SparkSession.builder
    .appName("05_Model_Training")
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.sql.files.maxRecordsPerFile", "500000")
    .config("spark.local.dir", r"D:/spark_tmp")
    .getOrCreate()
)

start_ts = datetime.now()
start_time = time.time()

print("\nModel training started ...")

# Load the latest feature dataset
features_path = get_latest_features_path()
print(f"Reading features from: {features_path}")

df = spark.read.parquet(features_path)

# I need "features" and "label" to train the model
required_cols = {"features", "label"}
missing = required_cols - set(df.columns)
if missing:
    spark.stop()
    raise SystemExit(f"Missing required columns in features parquet: {missing}")

total_rows = df.count()
print(f"Total feature rows: {total_rows}")

# Train/test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
n_train = train_df.count()
n_test  = test_df.count()
print(f"Split -> train: {n_train} | test: {n_test}")

# Logistic Regression model setup
# I’m starting with basic settings (no regularization)
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxIter=30,
    regParam=0.0,
    elasticNetParam=0.0
)

# Fit the model
print("Fitting model ...")
model = lr.fit(train_df)
print("Model fitted.")

# Evaluate on test set
print("Evaluating on test set ...")
pred = model.transform(test_df).cache()

# Binary metric
e_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = float(e_auc.evaluate(pred))

# Multi-class style metrics (accuracy, f1, precision, recall)
e_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
e_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
e_wp  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
e_wr  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

acc = float(e_acc.evaluate(pred))
f1  = float(e_f1.evaluate(pred))
prec = float(e_wp.evaluate(pred))
rec  = float(e_wr.evaluate(pred))

print(f"Metrics -> acc={acc:.4f}, f1={f1:.4f}, prec={prec:.4f}, recall={rec:.4f}, auc={auc:.4f}")

# Save model
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join(ARTIFACTS_DIR, f"lr_model_{run_id}")
model.save(model_dir)
print(f"Model saved to: {model_dir}")

# Log training report 
duration_sec = time.time() - start_time

conf = spark.sparkContext.getConf()
spark_master   = conf.get("spark.master", "unknown")
driver_mem     = conf.get("spark.driver.memory", "default")
exec_mem       = conf.get("spark.executor.memory", "default")
shuffle_parts  = spark.conf.get("spark.sql.shuffle.partitions", "default")

header = [
    "run_id","start_ts","duration_sec","spark_master","driver_memory","executor_memory","shuffle_partitions",
    "features_path","rows_total","rows_train","rows_test",
    "model_path","acc","f1","precision","recall","auc"
]
row = [
    run_id, start_ts.strftime("%Y-%m-%d %H:%M:%S"), f"{duration_sec:.2f}", spark_master, driver_mem, exec_mem, shuffle_parts,
    features_path, total_rows, n_train, n_test,
    model_dir, f"{acc:.6f}", f"{f1:.6f}", f"{prec:.6f}", f"{rec:.6f}", f"{auc:.6f}"
]

write_header = not os.path.exists(REPORT_TRAIN)
with open(REPORT_TRAIN, "a", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(header)
    w.writerow(row)

print(f"Training report appended to: {REPORT_TRAIN}")
print(f"Total time: {duration_sec:.2f} sec")
print("Training completed.")
spark.stop()
