from pyspark.sql.functions import col, lag, unix_timestamp
from pyspark.sql.window import Window

def cast_timestamp(df):
    """
    Casts the string Timestamp column to a proper timestamp type.
    """
    return df.withColumn("Timestamp", col("Timestamp").cast("timestamp"))

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
    If amount received is 0, we avoid division by zero.
    """
    df_with_ratio = df.withColumn(
        "Amount_Ratio", 
        col("Amount Paid") / col("Amount Received")
    )
    # Impute potential null ratios (like division by zero) to 1.0
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
