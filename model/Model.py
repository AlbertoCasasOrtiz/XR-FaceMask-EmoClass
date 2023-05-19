import os
os.environ['TF_GPU_ALLOCATOR'] = 'memory_growth'

import gc
import shutil
import tensorflow

import numpy as np
import matplotlib as mpl

from keras import backend
from abc import ABC, abstractmethod
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from metrics.Metrics import calculate_metrics, plot_history, save_metrics


class Model(ABC):

    def __init__(self, folds, epochs, directory, optimizers=None, learning_rate=None):
        mpl.use('TkAgg')
        if optimizers is None:
            self.optimizers = ['SGD']
        else:
            self.optimizers = optimizers
        if learning_rate is None:
            self.learning_rate = [0.01]
        else:
            self.learning_rate = learning_rate

        self.folds = folds
        self.epochs = epochs
        self.dir = directory

    @abstractmethod
    def define_and_compile_model(self, optimizer, learning_rate):
        mpl.use('TkAgg')
        pass

    def train_and_evaluate(self, images, labels, images_test, labels_test, class_names):
        # Define cross-validation splits
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=42)

        # Remove previous results
        if os.path.exists(f"results/{self.dir}/"):
            shutil.rmtree(f"results/{self.dir}/")

        # Train and evaluate models
        for optimizer in self.optimizers:
            for learning_rate in self.learning_rate:
                # Store fold metrics.
                fold_metrics = []
                for fold_idx, (train_idx, test_idx) in enumerate(skf.split(images, labels)):
                    print(f"Training model for fold: {fold_idx + 1}")
                    print(f"Optimizer: {optimizer}")
                    print(f"Learning rate: {learning_rate}")

                    # Create dir to store results
                    fold_dir = f"results/{self.dir}/{optimizer}/learning_rate-{str(learning_rate)}/fold_{fold_idx + 1}"
                    if not os.path.exists(fold_dir):
                        os.makedirs(fold_dir)

                    # Define and compile model
                    model = self.define_and_compile_model(optimizer, learning_rate)

                    # Split data into train, dev, and test sets
                    train_images, train_labels = images[train_idx], labels[train_idx]
                    dev_images, dev_labels = images[test_idx], labels[test_idx]

                    # Apply one hot encoding.
                    train_labels_one_hot = to_categorical(train_labels)
                    dev_labels_one_hot = to_categorical(dev_labels)
                    test_labels_one_hot = to_categorical(labels_test)

                    # Print sizes for train-dev-test
                    print("Train Set Size:", len(train_labels_one_hot))
                    print("Dev Set Size:", len(dev_labels_one_hot))
                    print("Test Set Size:", len(test_labels_one_hot))

                    # Train model
                    history = model.fit(train_images, train_labels_one_hot,
                                        epochs=self.epochs,
                                        validation_data=(dev_images, dev_labels_one_hot), batch_size=16,
                                        verbose=1)

                    # Predict labels of test set.
                    test_pred_probs = model.predict(images_test)
                    test_pred_labels = np.argmax(test_pred_probs, axis=1)

                    # Calculate metrics.
                    metrics = calculate_metrics(labels_test, test_pred_labels, fold_idx, fold_dir, class_names)
                    fold_metrics.append(metrics)

                    # Plot history of fold.
                    plot_history(history, fold_idx, fold_dir)

                    # Save weights
                    model.save_weights(f'results/{self.dir}/{optimizer}/learning_rate-{str(learning_rate)}/'
                                       f'fold_{fold_idx + 1}/model_weights_fold{fold_idx + 1}.h5')

                    print()

                    # release unused memory
                    backend.clear_session()
                    del model
                    gc.collect()
                    tensorflow.compat.v1.reset_default_graph()

                    # write the metrics to a CSV file
                    save_metrics(f'{self.dir}/{optimizer}/learning_rate-{str(learning_rate)}/fold_{fold_idx + 1}', fold_metrics)
