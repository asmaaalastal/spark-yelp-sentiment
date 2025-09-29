# Load findspark so Python can work with Spark
import findspark
findspark.init(r"C:\bigdata\spark-4.0.0-bin-hadoop3")

# Import libraries for Spark and some extra tools
import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, trim, length, avg, min as smin, max as smax

# Define the input dataset path and runs directory
DATA_PATH = "../data/yelp_academic_dataset_review.json" 
RUNS_DIR = "../runs"

# Create a Spark session to run the validation job
spark = (SparkSession.builder
         .appName("02_Data_Validation")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())

# Load the Yelp dataset
print(f"\nReading: {DATA_PATH}")
df = spark.read.json(DATA_PATH, multiLine=False)

# Print schema to understand data types
print("\nSchema:")
df.printSchema()

# Make sure that we have the important columns ("text" and "stars")
required = {"text", "stars"}
missing = required - set(df.columns)
if missing:
    # If columns are missing, stop the process
    print(f"\nMissing required columns: {missing}")
    print("Please fix data before proceeding.")
    spark.stop(); sys.exit(1)
else:
    print("\nFound required columns: text, stars")

# Count the total rows in the dataset
total_rows = df.count()
print(f"\nTotal rows: {total_rows}")

# Check for invalid values in stars column (values not in 1–5)
invalid_stars_cnt = df.filter(~col("stars").isin(1,2,3,4,5)).count()
print(f"Rows with stars NOT in [1..5]: {invalid_stars_cnt}")

# Count null values for "stars" and "text"
stars_nulls = df.filter(col("stars").isNull()).count()
text_nulls  = df.filter(col("text").isNull()).count()
print(f"Null stars: {stars_nulls}")
print(f"Null text : {text_nulls}")

# Count rows where text exists but is blank or only spaces
blank_texts = df.filter((col("text").isNotNull()) & (length(trim(col("text"))) == 0)).count()
print(f"Blank/whitespace-only text rows: {blank_texts}")

# Calculate text length statistics (min, avg, max)
len_stats = (df
    .filter(col("text").isNotNull())
    .select(length(col("text")).alias("len"))
    .agg(smin("len").alias("min_len"),
         avg("len").alias("avg_len"),
         smax("len").alias("max_len"))
).collect()[0]
print(f"Text length stats -> min: {len_stats['min_len']}, avg: {len_stats['avg_len']:.2f}, max: {len_stats['max_len']}")

# Check if review_id exists and find duplicate IDs if any
dup_info = "review_id not present"
if "review_id" in df.columns:
    dup_cnt = (df.groupBy("review_id").agg(count("*").alias("cnt"))
                 .filter(col("cnt") > 1).count())
    dup_info = f"duplicate review_id values: {dup_cnt}"
print(f"[*] Duplicate check -> {dup_info}")

# Show the distribution of star ratings
from pyspark.sql.functions import expr
stars_dist = (df.groupBy("stars").agg(count("*").alias("cnt")).orderBy("stars"))
stars_dist.show(truncate=False)

# Prepare planned binary labeling: 
# Negative (1,2), Neutral (3), Positive (4,5)
dist = {row['stars']: row['cnt'] for row in stars_dist.collect()}
neg = dist.get(1.0,0) + dist.get(2.0,0)
neu = dist.get(3.0,0)
pos = dist.get(4.0,0) + dist.get(5.0,0)
kept = neg + pos
pos_pct = (pos/kept*100) if kept>0 else 0
neg_pct = (neg/kept*100) if kept>0 else 0

print("\nPlanned binary labeling (drop 3-star):")
print(f"    Negative (1★+2★): {neg}")
print(f"    Positive (4★+5★): {pos}")
print(f"    Neutral  (3★)   : {neu}  (to be removed)")
print(f"    After drop -> kept: {kept}, Positive%: {pos_pct:.2f}%, Negative%: {neg_pct:.2f}%")

# Save validation results into a CSV report
report_path = os.path.join(RUNS_DIR, "validation_report.csv")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("metric,value\n")
    f.write(f"total_rows,{total_rows}\n")
    f.write(f"invalid_stars,{invalid_stars_cnt}\n")
    f.write(f"stars_nulls,{stars_nulls}\n")
    f.write(f"text_nulls,{text_nulls}\n")
    f.write(f"blank_texts,{blank_texts}\n")
    f.write(f"text_min_len,{len_stats['min_len']}\n")
    f.write(f"text_avg_len,{len_stats['avg_len']:.2f}\n")
    f.write(f"text_max_len,{len_stats['max_len']}\n")
    f.write(f"dup_review_id,{dup_info if isinstance(dup_info, str) else dup_cnt}\n")
    f.write(f"label_neg_planned,{neg}\n")
    f.write(f"label_pos_planned,{pos}\n")
    f.write(f"label_neu_planned,{neu}\n")
    f.write(f"label_pos_pct_planned,{pos_pct:.4f}\n")
    f.write(f"label_neg_pct_planned,{neg_pct:.4f}\n")

print(f"\nWrote validation report to: {report_path}")
print("Data validation completed. You can proceed to labeling.")

# Stop Spark session at the end
input("\nPress Enter to close...")
spark.stop()
