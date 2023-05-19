# XR-FaceMask-EmoClass

This is the source code accompanying our paper "Exploring the Impact of Partial Occlusion on Emotion Classifi-cation from Facial Expressions: A Comparative Study of XR Headsets and Face Masks".

## Instructions

This code has been tested on Windows, but minimal modifications should make it work on Linux and Mac environments.

1. Put your raw images into `"assets/raw_dataset/aligned/"`.
2. Put labels file into `"assets/raw_dataset/list_partition_label.txt"`.
  - Format: One line per instance.
    - `train_<number>.jpg <class>` for train instances.`
    - `test_<number>.jpg <class>` for test instances.`
3. Set `rebuild_dataset = True` if you want to generate the VR and Masked datasets.
4. Execute `main.py`.

### RAF-DB dataset

Access to the RAF-DB dataset can be requested here: http://www.whdeng.cn/raf/model1.html

If you are using the RAF-DB dataset, just put the aligned images directly into `"assets/raw_dataset/aligned/"` and the lables file into `"assets/raw_dataset/list_partition_label.txt"`.

### Data augmentation

The method used to frontalize and de-occlude the faces for data augmentation is called CFR-GAN, and can be found here: https://github.com/yeongjoonJu/CFR-GAN

We used CFR-GAN over the train set, and then manually copied the de-occluded and frontalized instances of the minoritary classes into the RAF-DB dataset to augment it.

### GPU training

If you want to train using GPU (way faster) on Windows, you can refer to the following links. Instructions should be similar for Linux and Mac environments:
1. https://medium.com/@ashkan.abbasi/quick-guide-for-installing-python-tensorflow-and-pycharm-on-windows-ed99ddd9598
2. https://discuss.tensorflow.org/t/tensorflow-gpu-not-working-on-windows/13120/3

### Inference and trained models

Coming soon...

### Notes

Ideal image size is (224, 224, 3). Please, resize your images to this size.

## Citation

> A. Casas-Ortiz, J. Echeverria, N. Jimenez-Tellez, y O. C. Santos, “Exploring the Impact of Partial Occlusion on Emotion Classifi-cation from Facial Expressions: A Comparative Study of XR Headsets and Face Masks”, 2023.
