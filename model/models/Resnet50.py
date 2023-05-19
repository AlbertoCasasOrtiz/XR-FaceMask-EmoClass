import tensorflow as tf

from model.Model import Model
from keras_vggface.vggface import VGGFace


class Resnet50(Model):

    def __init__(self, num_classes, image_size, folds, epochs, directory, optimizers, learning_rates):
        super().__init__(folds, epochs, directory, optimizers, learning_rates)

        self.num_classes = num_classes
        self.image_size = image_size

    def define_and_compile_model(self, optimizer, learning_rate):
        model = tf.keras.Sequential()

        model.add(VGGFace(model='resnet50', input_shape=self.image_size, include_top=False))
        model.add(tf.keras.layers.Flatten(name='flatten'))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        if optimizer == "SGD":
            optimizer_object = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == "RMSprop":
            optimizer_object = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == "Adagrad":
            optimizer_object = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer == "Adadelta":
            optimizer_object = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        elif optimizer == "Adam":
            optimizer_object = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "Adamax":
            optimizer_object = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
        elif optimizer == "Nadam":
            optimizer_object = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        else:
            raise Exception(f"Provided optimizer {optimizer} is not valid. Valid optimizers are: [SGD, RMSprop, "
                            f"Adagrad, Adadelta, Adam, Adamax, Nadam]")

        model.compile(optimizer=optimizer_object, loss="categorical_crossentropy", metrics=['accuracy'])
        print(model.summary())

        return model
