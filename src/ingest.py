import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

def get_spark_session(app_name="Anti Money Laundering Detection System"):
    """
    Creates and configures a PySpark SparkSession.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

def get_transaction_schema():
    """
    Returns the StructType schema for the IBM AML transaction dataset.
    """
    return StructType([
        StructField("Timestamp", StringType(), True),
        StructField("From Bank", IntegerType(), True),
        StructField("From Bank Hex_Code", StringType(), True),
        StructField("To Bank", IntegerType(), True),
        StructField("To Bank Hex_Code", StringType(), True),
        StructField("Amount Received", DoubleType(), True),
        StructField("Receiving Currency", StringType(), True),
        StructField("Amount Paid", DoubleType(), True),
        StructField("Payment Currency", StringType(), True),
        StructField("Payment Format", StringType(), True),
        StructField("Is Laundering", IntegerType(), True)
    ])

def load_data(spark, file_path):
    """
    Loads transaction data from a CSV file using Spark with the predefined schema.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")
        
    schema = get_transaction_schema()
    return spark.read.schema(schema) \
        .option("header", "True") \
        .option("timestampFormat", "yyyy/MM/dd HH:mm") \
        .csv(file_path)
