# 04_feature_engineering.py
# Goal: turn review text into numeric features (TF-IDF on unigrams + bigrams)
# So the model in the next step can actually learn from the text.

import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    NGram,
    CountVectorizer,
    IDF,
    VectorAssembler,
)

# I keep a run_id just to version the outputs (easier to track experiments)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Inputs/outputs: read labeled data, write features + save the fitted FE pipeline
IN_PARQUET  = "../runs/03_labeled_*.parquet"
OUT_PARQUET = f"../runs/04_features_{run_id}.parquet"
OUT_MODEL   = f"../artifacts/fe_pipeline_{run_id}"
OUT_REPORT  = "../runs/feature_report.csv"

# Make sure Spark has a temp dir (Windows likes this)
os.makedirs(r"D:\spark_tmp", exist_ok=True)

# Start Spark with reasonable defaults for my laptop
spark = (
    SparkSession.builder
    .appName("04_Feature_Engineering (Text -> Features)")
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.local.dir", r"D:/spark_tmp")
    .getOrCreate()
)

# Load the labeled dataset
print(f"\nReading labeled data from: {IN_PARQUET}")
df = spark.read.parquet(IN_PARQUET)

# Quick sanity check: I need 'text' and 'label' for FE
required = {"text", "label"}
missing = required - set(df.columns)
if missing:
    print(f"[!] Missing required columns: {missing}. Abort.")
    spark.stop()
    raise SystemExit(1)

# Just for the report
total_rows = df.count()
print(f"Total rows in labeled input: {total_rows}")

# Light text cleaning (I’m keeping it simple/cheap)
# - remove URLs
# - lowercase
# - collapse extra spaces
df_clean = (
    df.withColumn("text_nourl", regexp_replace(col("text"), r"http\S+|www\.\S+", " "))
      .withColumn("text_lower", lower(col("text_nourl")))
      .withColumn("text_clean", trim(regexp_replace(col("text_lower"), r"\s+", " ")))
)

# Tokenization + stopwords
# Split on non-word characters; I require token length >= 2 to drop tiny tokens
tokenizer = RegexTokenizer(
    inputCol="text_clean",
    outputCol="tokens",
    pattern=r"\W+",
    minTokenLength=2,
    toLowercase=False  # already lowercased above
)

# Remove common English stopwords to reduce noise
stopper = StopWordsRemover(
    inputCol="tokens",
    outputCol="tokens_nostop",
    caseSensitive=False
)

# Add bigrams (n=2) 
# This captures short phrases like "not good", which can help sentiment
bigrams = NGram(n=2, inputCol="tokens_nostop", outputCol="bigrams")

# Count features (bag-of-words) for unigrams + bigrams 
# Limit vocab size and require minDF to avoid super rare terms
cv_uni = CountVectorizer(
    inputCol="tokens_nostop",
    outputCol="tf_uni",
    vocabSize=50000,
    minDF=5
)
cv_bi = CountVectorizer(
    inputCol="bigrams",
    outputCol="tf_bi",
    vocabSize=50000,
    minDF=5
)

# TF-IDF weighting 
# IDF down-weights terms that appear everywhere (less informative)
idf_uni = IDF(inputCol="tf_uni", outputCol="tfidf_uni")
idf_bi  = IDF(inputCol="tf_bi",  outputCol="tfidf_bi")

# Final feature vector
# Concatenate unigram + bigram TF-IDF into one 'features' column
assembler = VectorAssembler(
    inputCols=["tfidf_uni", "tfidf_bi"],
    outputCol="features"
)

# Build the pipeline
# I like to keep everything in a Pipeline so I can save/reuse it later
pipeline = Pipeline(stages=[
    tokenizer,
    stopper,
    bigrams,
    cv_uni, cv_bi,
    idf_uni, idf_bi,
    assembler
])

# Fit + transform 
print("Fitting feature pipeline...")
model = pipeline.fit(df_clean)  # learns vocabularies + IDF stats

print("Transforming dataset to add 'features'...")
df_feat = model.transform(df_clean)

# Keep only the columns I actually need in the next steps
keep_cols = [c for c in ["review_id", "user_id", "business_id", "date", "stars", "label", "text", "features"] if c in df_feat.columns]
df_out = df_feat.select(*keep_cols)

# Grab vocab sizes just to report them (nice for the paper)
cv_uni_model = model.stages[3]
cv_bi_model  = model.stages[4]
vocab_uni = len(cv_uni_model.vocabulary)
vocab_bi  = len(cv_bi_model.vocabulary)

print(f"Unigram vocab size: {vocab_uni}")
print(f"Bigram  vocab size: {vocab_bi}")

# Save the feature dataset
print(f"Writing features dataset to: {OUT_PARQUET}")
(df_out
    .write
    .mode("overwrite")
    .parquet(OUT_PARQUET)
)

# Save the fitted pipeline (so training/inference use the same FE)
print(f"Saving fitted pipeline model to: {OUT_MODEL}")
if os.path.exists(OUT_MODEL):
    import shutil
    shutil.rmtree(OUT_MODEL)
model.save(OUT_MODEL)

# Append a small CSV report with run metadata
print(f"Appending feature report to: {OUT_REPORT}")
header_needed = not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), OUT_REPORT))) and not os.path.exists(OUT_REPORT)
with open(OUT_REPORT, "a", encoding="utf-8") as f:
    if header_needed:
        f.write("run_id,total_rows,vocab_uni,vocab_bi,out_parquet,out_model\n")
    f.write(f"{run_id},{total_rows},{vocab_uni},{vocab_bi},{OUT_PARQUET},{OUT_MODEL}\n")

print("Feature engineering completed.")
spark.stop()
