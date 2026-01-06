"""Evaluation utilities."""

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def save_comparison_report(baseline_metrics, hybrid_metrics, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({
        "Metric": baseline_metrics.keys(),
        "Baseline (DistilBERT)": baseline_metrics.values(),
        "Hybrid (DeBERTa + features)": hybrid_metrics.values()
    })
    df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    print(df)

def plot_confusion_matrices(y_true, baseline_pred, hybrid_pred, output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(confusion_matrix(y_true, baseline_pred), annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title('Baseline Confusion Matrix')
    
    sns.heatmap(confusion_matrix(y_true, hybrid_pred), annot=True, fmt='d', ax=axes[1], cmap='Greens')
    axes[1].set_title('Hybrid Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png")
    plt.close()