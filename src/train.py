import argparse
import os
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

from ingest import get_spark_session, load_data
from features import run_feature_pipeline

def build_pipeline(stages=None):
    """
    Builds the PySpark ML Pipeline stages for preprocessing and model training.
    """
    if stages is None:
        stages = []
        
    categorical_columns = ['Receiving Currency', 'Payment Currency', 'Payment Format']
    
    # Categorical preprocessing stages
    for col_name in categorical_columns:
        string_indexer = StringIndexer(inputCol=col_name, outputCol=col_name + 'Index', handleInvalid="keep")
        encoder = OneHotEncoder(inputCols=[string_indexer.getOutputCol()], outputCols=[col_name + 'classVec'])
        stages += [string_indexer, encoder]
        
    # Numerical features (including engineered ones)
    numeric_features = [
        "From Bank", "To Bank", "Amount Received", "Amount Paid",
        "Transaction_Duration", "Amount_Ratio", "Transaction_Frequency"
    ]
    
    # Assemble all features
    assembler_inputs = [c + "classVec" for c in categorical_columns] + numeric_features
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")
    stages += [assembler]
    
    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
    stages += [scaler]
    
    # Random Forest Classifier
    rf = RandomForestClassifier(labelCol="Is Laundering", featuresCol="scaledFeatures")
    stages += [rf]
    
    return Pipeline(stages=stages)

def main():
    parser = argparse.ArgumentParser(description="Train PySpark ML Random Forest model for AML Detection")
    parser.add_argument("--data_path", type=str, default="data/HI-Medium_Trans.csv", help="Path to raw transaction CSV file")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory where trained model will be saved")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of the dataset to sample for training (e.g. 0.05 for 5%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting and sampling")
    
    args = parser.parse_args()
    
    # 1. Initialize Spark session
    spark = get_spark_session()
    print("Spark Session created successfully.")
    
    # 2. Load data
    print(f"Loading data from {args.data_path}...")
    df = load_data(spark, args.data_path)
    
    # 3. Sample data if requested (useful for testing or resource constraints)
    if args.sample_fraction < 1.0:
        print(f"Sampling dataset with fraction: {args.sample_fraction}...")
        df = df.sample(withReplacement=False, fraction=args.sample_fraction, seed=args.seed)
        
    # 4. Feature Engineering
    print("Running feature engineering pipeline...")
    df_engineered = run_feature_pipeline(df)
    
    # 5. Split train/test sets (50/50 as done in the notebook for Random Forest)
    print("Splitting dataset into train and test sets...")
    train_data, test_data = df_engineered.randomSplit([0.5, 0.5], seed=args.seed)
    
    # Cache data to optimize fitting speed
    train_data.cache()
    
    # 6. Build and fit pipeline
    print("Building Spark ML Pipeline...")
    pipeline = build_pipeline()
    
    print("Fitting model...")
    pipeline_model = pipeline.fit(train_data)
    
    # 7. Save pipeline model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "rf_pipeline_model")
    print(f"Saving model to {model_path}...")
    pipeline_model.write().overwrite().save(model_path)
    
    # Optionally save test partition for separate evaluation
    test_path = os.path.join(args.model_dir, "test_data.parquet")
    print(f"Saving test set to parquet at {test_path}...")
    test_data.write.mode("overwrite").parquet(test_path)
    
    print("Training process completed successfully.")

if __name__ == "__main__":
    main()
