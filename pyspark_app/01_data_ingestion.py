# First, I load the findspark library so I can connect Python with Spark
import findspark
findspark.init(r"C:\bigdata\spark-4.0.0-bin-hadoop3")

# Import the needed Spark libraries for building a session and working with data
from pyspark.sql import SparkSession
from pyspark.sql.functions import count 

# Here I create a SparkSession, which is the entry point for any Spark job
spark = (SparkSession.builder
         .appName("Yelp Reviews Analysis")        # give the app a name
         .master("local[*]")                       # run locally using all available CPU cores
         .config("spark.sql.shuffle.partitions", "8")  # set number of shuffle partitions
         .getOrCreate())

# Load the Yelp reviews dataset from a JSON file
df = spark.read.json("../data/yelp_academic_dataset_review.json", multiLine=False)

# Print the list of columns and their count
print("\nColumns list:")
print(df.columns)
print("Number of columns:", len(df.columns))

# Print the schema (column names and data types)
print("\nSchema:")
df.printSchema()

# Select some familiar columns to display as a sample
show_cols = [c for c in ["stars", "text", "date", "user_id", "business_id"] if c in df.columns]
if show_cols:
    print("\nSample rows (5):")
    df.select(*show_cols).show(5, truncate=100)   # show 5 rows with limited text
else:
    print("\nDid not find familiar columns (stars/text/...). Check the schema above.")

# If the "stars" column exists, show the distribution of ratings
if "stars" in df.columns:
    print("\nStars distribution:")
    (df.groupBy("stars")
       .agg(count("*").alias("cnt"))
       .orderBy("stars")
       .show(truncate=False))
else:
    print("\nColumn 'stars' not found. Cannot compute distribution.")

# Optional: calculate the total number of rows (expensive operation, so default is False)
DO_FULL_COUNT = False
if DO_FULL_COUNT:
    df.cache()                      # cache the dataset in memory
    total_rows = df.count()         # count all rows
    print("\nTotal rows:", total_rows)
    df.unpersist()                  # remove from memory after counting

# Pause the program until I press Enter, then stop Spark
input("\nPress Enter to close...")
spark.stop()    # stop the SparkSession
