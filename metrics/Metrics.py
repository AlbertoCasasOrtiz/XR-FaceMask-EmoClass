import csv

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


def entropy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n = np.sum(cm)
    cm_norm = cm / n
    cm_norm[cm_norm == 0] = 1  # Avoid taking the log of 0
    return -np.sum(cm_norm * np.log2(cm_norm))


def calculate_metrics(test_labels, test_pred_labels, fold_idx, fold_dir, class_names):
    # Calculate metrics
    acc = accuracy_score(test_labels, test_pred_labels)
    ent = entropy(test_labels, test_pred_labels)
    f1_m = f1_score(test_labels, test_pred_labels, average='macro', zero_division=0)
    prec_m = precision_score(test_labels, test_pred_labels, average='macro', zero_division=0)
    rec_m = recall_score(test_labels, test_pred_labels, average='macro', zero_division=0)
    f1_w = f1_score(test_labels, test_pred_labels, average='weighted', zero_division=0)
    prec_w = precision_score(test_labels, test_pred_labels, average='weighted', zero_division=0)
    rec_w = recall_score(test_labels, test_pred_labels, average='weighted', zero_division=0)
    cm = confusion_matrix(test_labels, test_pred_labels)

    # Create heatmap of confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted labels', fontsize=16)
    ax.set_ylabel('True labels', fontsize=16)
    ax.set_title('Confusion Matrix', fontsize=18)
    ax.xaxis.set_ticklabels(class_names, fontsize=14)
    ax.yaxis.set_ticklabels(class_names, fontsize=14, rotation=0)
    plt.savefig(f"{fold_dir}/confusion_matrix.png", dpi=300)
    plt.close()

    # Print metrics
    print(f"Metrics for fold {fold_idx + 1}:")
    print(f"\tAccuracy: {acc}")
    print(f"\tEntropy: {ent}")
    print(f"\tF1-score (macro): {f1_m}")
    print(f"\tPrecision (macro): {prec_m}")
    print(f"\tRecall (macro): {rec_m}")
    print(f"\tF1-score (weighted)': {f1_w}")
    print(f"\tPrecision (weighted)': {prec_w}")
    print(f"\tRecall (weighted)': {rec_w}")
    print(f"\tConfusion Matrix:\n {cm}")

    return [round(acc, 2), round(ent, 2),
            round(f1_m, 2), round(prec_m, 2), round(rec_m, 2),
            round(f1_w, 2), round(prec_w, 2), round(rec_w, 2)]


def plot_history(history, fold_idx, fold_dir):
    # plot the training history for the fold
    plt.plot(history.history['loss'], label='train loss')
    print(history)
    plt.plot(history.history['val_loss'], label='test loss')
    plt.title(f"Fold {fold_idx + 1} - Model Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(f"{fold_dir}/loss_plot.png")
    plt.close()

    # plot the accuracy history for the fold
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='test acc')
    plt.title(f"Fold {fold_idx + 1} - Model Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig(f"{fold_dir}/accuracy_plot.png")
    plt.close()


def save_metrics(directory, fold_metrics):
    # write the metrics to a CSV file
    with open(f"results/{directory}/fold_metrics.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Fold', 'Accuracy', 'Entropy', 'F1-score (macro)', 'Precision (macro)', 'Recall (macro)',
                         'F1-score (weighted)', 'Precision (weighted)', 'Recall (weighted)'])
        for i, fold in enumerate(fold_metrics):
            writer.writerow([i + 1] + fold)
