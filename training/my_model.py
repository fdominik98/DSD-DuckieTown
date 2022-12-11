import tensorflow as tf
from keras.layers import Conv2D, Activation, Lambda, Flatten, Dense, MaxPooling2D,TimeDistributed,LSTM
from tensorflow.python.keras.layers import Dropout


class NewModel:
    @staticmethod
    def build_linear_branch(inputs=(15, 150, 200, 3)):
        # ? Input Normalization

        x = Lambda(lambda x: x / 255.0)(inputs)
        x = TimeDistributed(MaxPooling2D((2, 2), padding="valid"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        x = TimeDistributed(Conv2D(32, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(32, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(32, (3, 3), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(32, (3, 3), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        x = TimeDistributed(MaxPooling2D((3, 3), padding="valid"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        # ? Flatten
        x = TimeDistributed(Flatten())(x)
        
        x = LSTM(64,return_sequences=True)(x)
        x = LSTM(32)(x)
        x = Dense(1024, kernel_initializer="normal", activation="relu")(x)
        x = Dense(512, kernel_initializer="normal", activation="relu")(x)
        x = Dense(64, kernel_initializer="normal", activation="relu")(x)
        x = Dense(1, kernel_initializer="normal", name="Linear")(x)

        return x

    @staticmethod
    def build_angular_branch(inputs=(15, 150, 200, 3)):
        # ? Input Normalization
        x = Lambda(lambda x: x / 255.0)(inputs)
        x = TimeDistributed(MaxPooling2D((2, 2), padding="valid"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        x = TimeDistributed(Conv2D(32, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(32, (5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(32, (3, 3), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(32, (3, 3), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        x = TimeDistributed(MaxPooling2D((3, 3), padding="valid"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        
        # ? Flatten
        x = TimeDistributed(Flatten())(x)

        x = LSTM(64,return_sequences=True)(x)
        x = LSTM(32)(x)
        x = Dense(1024, kernel_initializer="normal", activation="relu")(x)
        x = Dense(512, kernel_initializer="normal", activation="relu")(x)
        x = Dense(64, kernel_initializer="normal", activation="relu")(x)
        x = Dense(1, kernel_initializer="normal", name="Angular")(x)

        return x

    @staticmethod
    def build(timestep=15, width=150, height=200):
        input_shape = (timestep, height, width, 3)
        inputs = tf.keras.Input(shape=input_shape)
        linearVelocity = NewModel.build_linear_branch(inputs)
        angularVelocity = NewModel.build_angular_branch(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=[linearVelocity, angularVelocity], name="MyModel")

        return model


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
    def build(width=200, height=150):
        input_shape = (height, width, 3)
        inputs = tf.keras.Input(shape=input_shape)
        linearVelocity = MyModel.build_linear_branch(inputs)
        angularVelocity = MyModel.build_angular_branch(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=[linearVelocity, angularVelocity], name="MyModel")

        return model
