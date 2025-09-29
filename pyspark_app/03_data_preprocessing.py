# Connect Python with Spark
import findspark
findspark.init(r"C:\bigdata\spark-4.0.0-bin-hadoop3")

# Regular imports
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, trim, length

# I/O paths (input JSON, timestamped Parquet output, and a small CSV report) 
DATA_PATH = "../data/yelp_academic_dataset_review.json"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PARQUET = f"../runs/03_labeled_{run_id}.parquet"
OUT_REPORT  = "../runs/labeling_report.csv"

# Make sure Spark temp dir exists (helps with large shuffles on Windows)
os.makedirs(r"D:\spark_tmp", exist_ok=True)

# Build Spark session with settings that are friendly for bigger data on a laptop 
spark = (
    SparkSession.builder
      .appName("03_Data_Preprocessing (Labeling)")            # this job = labeling step
      .config("spark.sql.shuffle.partitions", "64")           # number of shuffle partitions
      .config("spark.sql.files.maxRecordsPerFile", "300000")  # smaller parquet files
      .config("spark.local.dir", r"D:/spark_tmp")             # local scratch space for Spark
      # pin Python executables (important when running via spark-submit on Windows)
      .config("spark.pyspark.python", r"C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe")
      .config("spark.pyspark.driver.python", r"C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe")
      .getOrCreate()
)

# Load raw Yelp JSON 
print(f"\nReading: {DATA_PATH}")
df = spark.read.json(DATA_PATH, multiLine=False)

#  Sanity check: must have 'text' and 'stars' to continue 
required = {"text", "stars"}
missing = required - set(df.columns)
if missing:
    print(f"Missing required columns: {missing}. Abort.")
    spark.stop(); raise SystemExit(1)

# Count rows before any filtering (for the report)
total_before = df.count()
print(f"Total rows (raw): {total_before}")

# Minimal cleaning: drop null/blank texts 
df = df.filter(col("text").isNotNull())
df = df.filter(length(trim(col("text"))) > 0)

# Create a binary label from star ratings 
# rule: 1-2 => 0 (negative), 4-5 => 1 (positive), 3 => drop (neutral)
df = df.withColumn(
    "label",
    when(col("stars").isin(4, 5), 1.0)
     .when(col("stars").isin(1, 2), 0.0)
     .otherwise(None)
).filter(col("label").isNotNull())

# Keep only useful columns for downstream steps
keep_cols = [c for c in ["review_id","user_id","business_id","date","stars","text","label"] if c in df.columns]
df = df.select(*keep_cols)

# Count rows after removing neutrals/empty texts (for the report)
total_after = df.count()
print(f"Total rows after dropping neutral (3-star) & empty text: {total_after}")

# Show label distribution so we can see class imbalance clearly
print("\nLabel distribution (0=negative, 1=positive):")
df.groupBy("label").agg(count("*").alias("cnt")).orderBy("label").show(truncate=False)

# Collect counts for the CSV report
dist = {r["label"]: r["cnt"] for r in df.groupBy("label").agg(count("*").alias("cnt")).collect()}
neg, pos = int(dist.get(0.0,0)), int(dist.get(1.0,0))
kept = neg + pos
neg_pct = (neg/kept*100) if kept else 0.0
pos_pct = (pos/kept*100) if kept else 0.0

# Write labeled dataset to Parquet (partitioned by label)
print(f"Saving labeled dataset to: {OUT_PARQUET}")
df = df.repartition(64, "label")  # helps create balanced files per class

(df.write
   .mode("overwrite")
   .option("maxRecordsPerFile", 300000)
   .partitionBy("label")
   .parquet(OUT_PARQUET))

# Write a tiny CSV report with key numbers (useful for the paper)
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    f.write(f"total_before,{total_before}\n")
    f.write(f"total_after,{total_after}\n")
    f.write(f"label_neg,{neg}\n")
    f.write(f"label_pos,{pos}\n")
    f.write(f"label_neg_pct,{neg_pct:.4f}\n")
    f.write(f"label_pos_pct,{pos_pct:.4f}\n")

print(f"Wrote labeling report to: {OUT_REPORT}")
print(f"Labeled Parquet written at: {OUT_PARQUET}")
spark.stop()
