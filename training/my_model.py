import tensorflow as tf
from keras.layers import Conv2D, Activation, Lambda, Flatten, Dense, MaxPooling2D
from tensorflow.python.keras.layers import Dropout

class MyModel:
    @staticmethod
    def build_linear_branch(inputs=(150, 200, 3)):
        # ? Input Normalization
        x = Lambda(lambda x: x / 255.0)(inputs)
        x = MaxPooling2D((2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        x = Conv2D(42, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="valid", kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="valid", kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((3, 3), padding="valid")(x)
        x = Activation("relu")(x)

        # ? Flatten
        x = Flatten()(x)

        x = Dense(1164, kernel_initializer="normal", activation="relu")(x)
        x = Dense(100, kernel_initializer="normal", activation="relu")(x)
        x = Dense(50, kernel_initializer="normal", activation="relu")(x)
        x = Dense(10, kernel_initializer="normal", activation="relu")(x)
        x = Dense(1, kernel_initializer="normal", name="Linear")(x)

        return x

    @staticmethod
    def build_angular_branch(inputs=(150, 200, 3)):
        # ? Input Normalization
        x = Lambda(lambda x: x / 255.0)(inputs)
        x = MaxPooling2D((2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        x = Conv2D(32, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        x = Conv2D(42, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="valid", kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), padding="valid", kernel_initializer="he_normal")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((3, 3), padding="valid")(x)
        x = Activation("relu")(x)

        # ? Flatten
        x = Flatten()(x)

        x = Dense(1164, kernel_initializer="normal", activation="relu")(x)
        x = Dense(100, kernel_initializer="normal", activation="relu")(x)
        x = Dense(50, kernel_initializer="normal", activation="relu")(x)
        x = Dense(10, kernel_initializer="normal", activation="relu")(x)
        x = Dense(1, kernel_initializer="normal", name="Angular")(x)

        return x

    @staticmethod
    def build(width=150, height=200):
        input_shape = (height, width, 3)
        inputs = tf.keras.Input(shape=input_shape)
        linearVelocity = MyModel.build_linear_branch(inputs)
        angularVelocity = MyModel.build_angular_branch(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=[linearVelocity, angularVelocity], name="MyModel")

        return model
