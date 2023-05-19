import os
import random
import shutil

import matplotlib as mpl

from model.Model import Model
from sklearn.model_selection import StratifiedKFold
from metrics.Metrics import calculate_metrics, save_metrics


class RandomClassifier(Model):

    def __init__(self, num_classes, folds, directory):
        super().__init__(folds, None, directory, None)
        mpl.use('TkAgg')

        self.num_classes = num_classes
        self.folds = folds
        self.dir = directory
        pass

    def define_and_compile_model(self, optimizer, learning_rate):
        # Not necessary in the random classifier.
        pass

    def train_and_evaluate(self, images, labels, images_test, labels_test, class_names):
        # Define cross-validation splits
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=42)

        # Store fold metrics.
        fold_metrics = []

        # Remove previous results
        if os.path.exists(f"results/{self.dir}/"):
            shutil.rmtree(f"results/{self.dir}/")

        # Train and evaluate models
        for fold_idx, (_, _) in enumerate(skf.split(images, labels)):
            print(f"Training model for fold {fold_idx + 1}")

            # Create dir to store results
            fold_dir = f"results/{self.dir}/fold_{fold_idx + 1}"
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            # Predict labels randomly for test set.
            test_pred_labels = self.predict(labels_test)

            metrics = calculate_metrics(labels_test, test_pred_labels, fold_idx, fold_dir, class_names)

            # append the metrics to the list
            fold_metrics.append(metrics)

            print()

        save_metrics(self.dir, fold_metrics)

    @staticmethod
    def predict(images):
        labels = []
        for _ in images:
            label = random.choice([0, 1, 2, 3, 4, 5, 6])  # replace with your own class names
            labels.append(label)
        return labels
