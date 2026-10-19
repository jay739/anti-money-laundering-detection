from pyspark.sql.functions import col, lag, lit, to_timestamp, unix_timestamp, when

# Format of the Timestamp column in the IBM AML dataset, e.g. 2022/09/01 00:20
TIMESTAMP_FORMAT = "yyyy/MM/dd HH:mm"
from pyspark.sql.window import Window

def cast_timestamp(df):
    """
    Parses the string Timestamp column into a proper timestamp type.

    A plain cast does not understand the dataset's yyyy/MM/dd HH:mm format:
    it returns NULL on Spark 3 and raises CAST_INVALID_INPUT on Spark 4.
    """
    return df.withColumn("Timestamp", to_timestamp(col("Timestamp"), TIMESTAMP_FORMAT))

def add_transaction_duration(df):
    """
    Calculates Transaction_Duration: the time difference in seconds between 
    consecutive transactions made to the same bank.
    """
    # Partition by the destination bank and order by time
    window_spec = Window.partitionBy("To Bank").orderBy("Timestamp")
    
    # Lag timestamp to get the previous transaction time
    df_with_prev = df.withColumn("Prev_Timestamp", lag("Timestamp").over(window_spec))
    
    # Calculate duration in seconds
    df_with_duration = df_with_prev.withColumn(
        "Transaction_Duration",
        (unix_timestamp(col("Timestamp")) - unix_timestamp(col("Prev_Timestamp")))
    )
    
    # Fill null values (first transaction for a bank) with 0
    return df_with_duration.fillna({"Transaction_Duration": 0}).drop("Prev_Timestamp")

def add_amount_ratio(df):
    """
    Calculates Amount_Ratio: Amount Paid divided by Amount Received.
    A zero amount received gives 1.0. The guard is explicit because Spark 4
    raises on division by zero instead of returning NULL.
    """
    df_with_ratio = df.withColumn(
        "Amount_Ratio",
        when(col("Amount Received") != 0, col("Amount Paid") / col("Amount Received"))
        .otherwise(lit(1.0)),
    )
    # Impute ratios left null by missing amounts to 1.0
    return df_with_ratio.fillna({"Amount_Ratio": 1.0})

def add_transaction_frequency(df):
    """
    Calculates Transaction_Frequency: total historical transaction count 
    for each sending bank ("From Bank").
    """
    # Calculate frequency counts
    freq_df = df.groupBy("From Bank").count().withColumnRenamed("count", "Transaction_Frequency")
    
    # Join back with the main dataframe
    return df.join(freq_df, on="From Bank", how="left")

def run_feature_pipeline(df):
    """
    Applies the full feature engineering pipeline.
    """
    df = cast_timestamp(df)
    df = add_transaction_duration(df)
    df = add_amount_ratio(df)
    df = add_transaction_frequency(df)
    return df
