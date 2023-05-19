import os
import cv2
import shutil
import fileinput

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from vrfilter.Filter import Filter


def check_dataset_exists(folders, label_file, label_test_file=None):
    exist_and_consistent = True

    # Check if label_file exists.
    if label_test_file is not None:
        if not os.path.isfile(label_test_file):
            exist_and_consistent = False
        else:
            # If label_file_exists, count number of labels.
            with fileinput.input(files=label_test_file) as f:
                line_count_test = 0
                for _ in f:
                    line_count_test += 1
    else:
        line_count_test = 0

    if not os.path.isfile(label_file):
        exist_and_consistent = False
    else:
        # If label_file_exists, count number of labels.
        with fileinput.input(files=label_file) as f:
            line_count = 0
            for _ in f:
                line_count += 1

        # Now, check that each folder has exactly the same number of elements as labels in label_file.
        for path in folders:
            # First, check that the folder exists.
            if os.path.exists(path):
                # If the folder exists, count number of images.
                num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                # If number of images is not the same as number of labels, the dataset is not consistent.
                if num_files != line_count + line_count_test:
                    exist_and_consistent = False
            # If the folder does not exist, the dataset is not consistent.
            else:
                exist_and_consistent = False

    # Check that every element in test
    return exist_and_consistent


def load_images_names_and_labels(label_file, one_based=False, add_aligned=False):
    img_names = []
    labels = []
    print("Loading labels...")
    with open(label_file, "r") as f:
        for line in f:
            img_name, label = line.strip().split()
            img_names.append(img_name)
            labels.append(int(label))
    labels = np.array(labels)
    if one_based:
        labels = labels - 1
    if add_aligned:
        img_names = [sub.replace('.jpg', '_aligned.jpg') for sub in img_names]
    print("Labels loaded:", len(labels))

    return img_names, labels


def recreate_dataset(img_folder_raw, label_file_raw, img_folder_orig, img_folder_filter_vr, img_folder_filter_face_mask,
                     label_file_new, label_test_file, new_size, crop=False):

    # Remove previous iteration's folders if exist.
    if os.path.exists(img_folder_filter_vr):
        shutil.rmtree(img_folder_filter_vr)
    if not os.path.exists(img_folder_filter_vr):
        os.makedirs(img_folder_filter_vr)
    if os.path.exists(img_folder_filter_face_mask):
        shutil.rmtree(img_folder_filter_face_mask)
    if not os.path.exists(img_folder_filter_face_mask):
        os.makedirs(img_folder_filter_face_mask)
    if os.path.exists(img_folder_orig):
        shutil.rmtree(img_folder_orig)
    if not os.path.exists(img_folder_orig):
        os.makedirs(img_folder_orig)
    if os.path.isfile(label_file_new):
        os.remove(label_file_new)
    if os.path.isfile(label_test_file):
        os.remove(label_test_file)

    # Load original images and labels
    print("Loading image names and labels...")
    img_names, labels = load_images_names_and_labels(label_file_raw, True, True)

    # Copy images into folder
    print("Copying images into folder...")
    dataset_raw_images_path = img_folder_raw
    for image_name in img_names:
        shutil.copy(dataset_raw_images_path + image_name, img_folder_orig)

    # Detect faces and crop/resize to 100x100.
    failed_image_names_detect_face = []
    if crop:
        print("Detecting faces and cropping/resizing...")
        filters = Filter()
        failed_image_names_detect_face = filters.apply_detect_face_crop_folder(img_folder_orig, img_folder_orig,
                                                                               new_size)

    # Create filtered vr images
    print("Creating VR filtered images...")
    filters = Filter()
    failed_image_names_vr = filters.apply_filter_vr_folder(img_folder_orig, img_folder_filter_vr)

    # Create filtered face mask images
    print("Creating face mask filtered images...")
    filters = Filter()
    failed_image_names_mask = filters.apply_filter_face_mask_folder(img_folder_orig, img_folder_filter_face_mask)

    # Union of failed.
    print("Merging failed results to remove...")
    failed_image_names = list(set(failed_image_names_detect_face)
                              .union(set(failed_image_names_vr))
                              .union(set(failed_image_names_mask)))

    print("Face was not recognized for", len(failed_image_names), "images. Removing...")

    # Remove failed elements from img_names and labels.
    print("Removing failed elements...")
    for image_name in failed_image_names:
        idx = img_names.index(image_name)
        del img_names[idx]
        labels = np.delete(labels, idx)

    # Save new labels file.
    print("Saving new labels...")
    new_labels = [f"{name} {num}" for name, num in zip(img_names, labels)]
    with open(label_file_new, "w") as f:
        with open(label_test_file, "w") as f_test:
            for i, item in enumerate(new_labels):
                if "train" in item:
                    f.write("%s\n" % item)
                elif "test" in item:
                    f_test.write("%s\n" % item)

    # Return new labels.
    return img_names, labels


def load_images(img_names, img_folder):
    # Load images
    images = []
    for img_name in img_names:
        img_path = img_folder + img_name
        img = plt.imread(img_path)
        img = img / 255.0
        resized_img = cv2.resize(img, (224, 224))
        images.append(resized_img)
    images = np.array(images)

    return images


def calculate_statistics(labels):
    label_counts = Counter(labels)

    # Calculate the percentage of each label
    total_count = len(labels)
    label_percentages = {}
    for label, count in label_counts.items():
        label_percentages[label] = count / total_count * 100

    return label_percentages
