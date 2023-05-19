from model.models.Vgg16 import Vgg16
from dataset.Dataset import load_images
from model.models.Senet50 import Senet50
from model.models.Resnet50 import Resnet50
from model.models.RandomClassifier import RandomClassifier


def train_and_evaluate(model_name, dataset_name,
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder, optimizer_list,
                       learning_rates,
                       folds=2, epochs=1,
                       image_shape=(224, 224, 3)):

    # Load orig images.
    print(f"Loading {dataset_name} images...")
    images = load_images(img_names, img_folder)
    images_test = load_images(img_names_test, img_folder)
    print(f"{dataset_name} images loaded:", len(images))
    print()

    print(f"Training {model_name} over {dataset_name} images...")
    if model_name == "RandomClassifier":
        # Define, compile and train a Random classifier over orig images.
        random_classifier = RandomClassifier(7, folds, f"{model_name}_{dataset_name}")
        random_classifier.train_and_evaluate(images, labels, images_test, labels_test,
                                             ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])

    elif model_name == "Vgg16":
        # Define, compile and train a CNN model over orig images
        cnn_model = Vgg16(7, image_shape, folds, epochs, f"{model_name}_{dataset_name}", optimizer_list, learning_rates)
        cnn_model.train_and_evaluate(images, labels, images_test, labels_test,
                                     ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])

    elif model_name == "Resnet50":
        # Define, compile and train a CNN model over orig images
        cnn_model = Resnet50(7, image_shape, folds, epochs, f"{model_name}_{dataset_name}", optimizer_list, learning_rates)
        cnn_model.train_and_evaluate(images, labels, images_test, labels_test,
                                     ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])

    elif model_name == "Senet50":
        # Define, compile and train a CNN model over orig images
        cnn_model = Senet50(7, image_shape, folds, epochs, f"{model_name}_{dataset_name}", optimizer_list, learning_rates)
        cnn_model.train_and_evaluate(images, labels, images_test, labels_test,
                                     ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])

    print(f"Model {model_name} over {dataset_name} images...")
    print()
