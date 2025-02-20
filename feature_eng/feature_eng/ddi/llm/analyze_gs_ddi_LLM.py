#
# The following script opens the results obtained using LLM and print confusion metrix and metrics
# 

import argparse
import pickle
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score,
                             matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns

def loadPickle(input_pickle):
    with open(input_pickle, 'rb') as f:
        r = pickle.load(f)
    return r

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

def main(results_pickle):
    res = loadPickle(results_pickle)

    # Calculate metrics
    true_labels = [d['target'] for d in res]
    predicted_labels = [d['new_target'] for d in res]

    cm = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, labels=[0, 1])
    recall = recall_score(true_labels, predicted_labels, labels=[0, 1]) # Sensitivity
    f1 = f1_score(true_labels, predicted_labels, labels=[0, 1])
    if len(set(true_labels)) > 1:
        roc_auc = roc_auc_score(true_labels, predicted_labels)
    else:
        roc_auc = None
        print("ROC AUC cannot be calculated due to lack of class diversity.")
    mcc = matthews_corrcoef(true_labels, predicted_labels)

    # Print metrics
    print("Confusion Matrix:\n", cm)
    print("\nAccuracy:", accuracy)
    print("Precision (PR):", precision)
    print("Sensitivity (SE):", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", roc_auc)
    print("Matthews Correlation Coefficient (MCC):", mcc)

    # Optional: full classification report
    report = classification_report(true_labels, predicted_labels)
    print("\nClassification Report:\n", report)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names=['no interaction', 'interaction'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis of the ddi classifications using LLMs.")
    parser.add_argument('result_pickle', type=str, help='Pickle file where the results have been saved.')
    
    args = parser.parse_args()
    main(args.result_pickle)