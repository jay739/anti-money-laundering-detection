import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from ingest import get_spark_session

def plot_curves(labels, probabilities, output_dir):
    """
    Plots and saves ROC and Precision-Recall curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    
    # 2. Precision-Recall Curve (Highly informative for imbalanced classes)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()
    
    print(f"ROC and PR curves saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate PySpark ML Random Forest model")
    parser.add_argument("--model_path", type=str, default="models/rf_pipeline_model", help="Path to trained pipeline model directory")
    parser.add_argument("--test_data_path", type=str, default="models/test_data.parquet", help="Path to test parquet dataset")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory where evaluation plots will be saved")
    
    args = parser.parse_args()
    
    spark = get_spark_session()
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Trained model not found at: {args.model_path}")
    if not os.path.exists(args.test_data_path):
        raise FileNotFoundError(f"Test data not found at: {args.test_data_path}")
        
    print(f"Loading model from {args.model_path}...")
    model = PipelineModel.load(args.model_path)
    
    print(f"Loading test data from {args.test_data_path}...")
    test_df = spark.read.parquet(args.test_data_path)
    
    print("Computing predictions...")
    predictions = model.transform(test_df)
    
    # Evaluate using Spark standard evaluators
    evaluator_roc = BinaryClassificationEvaluator(labelCol='Is Laundering', metricName='areaUnderROC')
    evaluator_acc = MulticlassClassificationEvaluator(labelCol='Is Laundering', metricName='accuracy')
    evaluator_prec = MulticlassClassificationEvaluator(labelCol='Is Laundering', metricName='weightedPrecision')
    evaluator_rec = MulticlassClassificationEvaluator(labelCol='Is Laundering', metricName='weightedRecall')
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol='Is Laundering', metricName='f1')
    
    roc_auc = evaluator_roc.evaluate(predictions)
    accuracy = evaluator_acc.evaluate(predictions)
    precision = evaluator_prec.evaluate(predictions)
    recall = evaluator_rec.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)
    
    print("\n" + "="*40)
    print("Spark ML Evaluation Metrics:")
    print("="*40)
    print(f"ROC AUC:    {roc_auc:.6f}")
    print(f"Accuracy:   {accuracy:.6f}")
    print(f"Precision:  {precision:.6f}")
    print(f"Recall:     {recall:.6f}")
    print(f"F1 Score:   {f1_score:.6f}")
    print("="*40 + "\n")
    
    # Extract probabilities and labels for matplotlib plotting
    print("Extracting prediction probabilities for plotting curves...")
    pred_data = predictions.select('probability', 'Is Laundering').collect()
    
    probabilities = [row['probability'][1] for row in pred_data]
    labels = [row['Is Laundering'] for row in pred_data]
    
    plot_curves(labels, probabilities, args.output_dir)

if __name__ == "__main__":
    main()
