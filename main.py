import tensorflow as tf

from dataset.Dataset import calculate_statistics
from model.ModelSelector import train_and_evaluate
from dataset.Dataset import recreate_dataset, load_images_names_and_labels

if __name__ == '__main__':
    # Print GPU information.
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if len(tf.config.list_physical_devices('GPU')) > 0:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print("Name:", gpu.name, "  Type:", gpu.device_type)
            tf.config.experimental.set_memory_growth(gpu, True)
    print()

    # Print CPU information.
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
    if len(tf.config.list_physical_devices('CPU')) > 0:
        cpus = tf.config.list_physical_devices('CPU')
        for cpu in cpus:
            print("Name:", cpu.name, "  Type:", cpu.device_type)
    print()

    # Image shape:
    image_shape = (224, 224, 3)
    image_shape_2 = image_shape[:2]

    # Paths to dataset folders.
    img_folder_raw = "assets/raw_dataset/original/aligned/"
    img_folder_filter_vr = "assets/dataset/images/filter_vr/"
    img_folder_filter_mask = "assets/dataset/images/filter_mask/"
    img_folder_orig = "assets/dataset/images/original/"

    # Paths to labels file.
    label_file_raw = "assets/raw_dataset/original/list_partition_label.txt"
    label_file = "assets/dataset/list_partition_label.txt"
    label_test_file = "assets/dataset/list_partition_label_dev.txt"

    # Check if dataset exists and is consistent.
    rebuild_dataset = True

    if rebuild_dataset:
        # Recreate dataset
        print("Recreating dataset...")
        _, labels = recreate_dataset(img_folder_raw, label_file_raw, img_folder_orig, img_folder_filter_vr,
                                     img_folder_filter_mask, label_file, label_test_file, image_shape_2, True)
        print("Dataset recreated:", len(labels))
        print()
    else:
        print("Dataset exists...")
        print()

    # Load training and test sets.
    img_names, labels = load_images_names_and_labels(label_file)
    img_names_test, labels_test = load_images_names_and_labels(label_test_file)

    # Calculate statistics
    label_percentages = calculate_statistics(labels)
    # Ordered as expected by the loop.
    label_names = ["Sadness", "Happiness", "Surprise", "Anger", "Fear", "Disgust", "Neutral"]
    # Print the label percentages
    i = 0
    for label, percentage in label_percentages.items():
        print(f"{label_names[i]}: {percentage:.2f}%")
        i = i + 1
    print()

    # Define optimizers.
    optimizer_list = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    learning_rates = [0.01, 0.001, 0.0001]
    optimizer_list = ['SGD', 'RMSprop',]
    learning_rates = [0.01, 0.001]

    # Train and evaluate models.
    train_and_evaluate("RandomClassifier", "Original",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_orig, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)

    train_and_evaluate("Senet50", "Original",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_orig, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)
    train_and_evaluate("Senet50", "VR",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_filter_vr, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)
    train_and_evaluate("Senet50", "Mask",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_filter_mask, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)

    train_and_evaluate("Resnet50", "Original",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_orig, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)
    train_and_evaluate("Resnet50", "VR",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_filter_vr, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)
    train_and_evaluate("Resnet50", "Mask",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_filter_mask, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)

    train_and_evaluate("Vgg16", "Original",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_orig, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)
    train_and_evaluate("Vgg16", "VR",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_filter_vr, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)
    train_and_evaluate("Vgg16", "Mask",
                       img_names, labels,
                       img_names_test, labels_test,
                       img_folder_filter_mask, optimizer_list,
                       learning_rates,
                       image_shape=image_shape)
