import tensorflow as tf
from keras.layers import Conv2D, Activation, Lambda, Flatten, Dense, MaxPooling2D,TimeDistributed,LSTM
from tensorflow.python.keras.layers import Dropout


class NewModel:
    @staticmethod
    def build_linear_branch(hp, inputs=(15, 150, 200, 3)):
        # ? Input Normalization

        x = Lambda(lambda x: x / 255.0)(inputs)
        x = TimeDistributed(MaxPooling2D((2, 2), padding="valid"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        x = TimeDistributed(Conv2D(
            # Tune number of units
            filters=hp.Int("linear_conv2d_filters", min_value=16, max_value=128, step=32), kernel_size=(5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(
            # Tune number of units
            filters=hp.Int("linear_conv2d_filters_2", min_value=16, max_value=128, step=32), kernel_size=(5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(
            # Tune number of units
            filters=hp.Int("linear_conv2d_filters_3", min_value=4, max_value=64, step=10), kernel_size=(3, 3), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(
            # Tune number of units
            filters=hp.Int("linear_conv2d_filters_4", min_value=4, max_value=64, step=10), kernel_size=(3, 3), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        x = TimeDistributed(MaxPooling2D((3, 3), padding="valid"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        # ? Flatten
        x = TimeDistributed(Flatten())(x)
        
        #x = LSTM(64,return_sequences=True)(x)
        x = LSTM(
            # Tune number of units
            units=hp.Int("linear_lstm_units", min_value=16, max_value=128, step=32),
            return_sequences=True)(x)
        #x = LSTM(32)(x)
        x = LSTM(
            # Tune number of units
            units=hp.Int("linear_lstm_units_2", min_value=16, max_value=64, step=16),
            return_sequences=True)(x)
        #x = Dense(1024, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("linear_units", min_value=512, max_value=1200, step=64),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("linear_activation", ["relu", "tanh"]))(x)
        #x = Dense(512, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("linear_units_2", min_value=206, max_value=712, step=64),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("linear_activation_2", ["relu", "tanh"]))(x)
        #x = Dense(64, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("linear_units_3", min_value=32, max_value=64, step=16),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("linear_activation_3", ["relu", "tanh"]))(x)
        x = Dense(1, kernel_initializer="normal", name="Linear")(x)

        return x

    @staticmethod
    def build_angular_branch(hp, inputs=(15, 150, 200, 3)):
        # ? Input Normalization
        x = Lambda(lambda x: x / 255.0)(inputs)
        x = TimeDistributed(MaxPooling2D((2, 2), padding="valid"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        x = TimeDistributed(Conv2D(
            # Tune number of units
            filters=hp.Int("angular_conv2d_filters", min_value=16, max_value=128, step=32), kernel_size=(5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(
            # Tune number of units
            filters=hp.Int("angular_conv2d_filters_2", min_value=16, max_value=128, step=32), kernel_size=(5, 5), strides=(2, 2), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(
            # Tune number of units
            filters=hp.Int("angular_conv2d_filters_3", min_value=4, max_value=64, step=10), kernel_size=(3, 3), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        x = TimeDistributed(Conv2D(
            # Tune number of units
            filters=hp.Int("angular_conv2d_filters_4", min_value=4, max_value=64, step=10), kernel_size=(3, 3), padding="valid", kernel_initializer="he_normal"))(x)
        x = TimeDistributed(Activation("relu"))(x)

        x = TimeDistributed(MaxPooling2D((3, 3), padding="valid"))(x)
        x = TimeDistributed(Activation("relu"))(x)
        
        # ? Flatten
        x = TimeDistributed(Flatten())(x)

        #x = LSTM(64,return_sequences=True)(x)
        x = LSTM(
            # Tune number of units
            units=hp.Int("angular_lstm_units", min_value=16, max_value=128, step=32),
            return_sequences=True)(x)
        #x = LSTM(32)(x)
        x = LSTM(
            # Tune number of units
            units=hp.Int("angular_lstm_units_2", min_value=16, max_value=64, step=16))(x)
        #x = Dense(1024, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("angular_units", min_value=512, max_value=1200, step=64),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("angular_activation", ["relu", "tanh"]))(x)
        #x = Dense(512, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("angular_units_2", min_value=206, max_value=712, step=64),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("angular_activation_2", ["relu", "tanh"]))(x)
        #x = Dense(64, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("angular_units_3", min_value=32, max_value=64, step=16),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("angular_activation_3", ["relu", "tanh"]))(x)
        x = Dense(1, kernel_initializer="normal", name="Angular")(x)

        return x

    @staticmethod
    def build(hp):
        timestep=4
        width=200
        height=150
        input_shape = (timestep, height, width, 3)
        inputs = tf.keras.Input(shape=input_shape)
        linearVelocity = NewModel.build_linear_branch(hp, inputs)
        angularVelocity = NewModel.build_angular_branch(hp, inputs)

        model = tf.keras.Model(inputs=inputs, outputs=[linearVelocity, angularVelocity], name="MyModel")

        losses = {"Linear": "mse", "Angular": "mse"}
        lossWeights = {"Linear": 2, "Angular": 10}
        init_lr = 1e-3
        opt = tf.keras.optimizers.Adam(init_lr)
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics="mse")

        return model


class MyModel:
    @staticmethod
    def build_linear_branch(hp, inputs=(150, 200, 3)):
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

        #x = Dense(1164, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("linear_units", min_value=512, max_value=1164, step=64),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("linear_activation", ["relu", "tanh"]))(x)
        #x = Dense(100, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("linear_units_2", min_value=50, max_value=200, step=32),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("linear_activation_2", ["relu", "tanh"]))(x)
        #x = Dense(50, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("linear_units_3", min_value=10, max_value=60, step=16),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("linear_activation_3", ["relu", "tanh"]))(x)
        #x = Dense(10, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("linear_units_4", min_value=2, max_value=10, step=2),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("linear_activation_4", ["relu", "tanh"]))(x)
        x = Dense(1, kernel_initializer="normal", name="Linear")(x)

        return x

    @staticmethod
    def build_angular_branch(hp, inputs=(150, 200, 3)):
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

        #x = Dense(1164, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("angular_units", min_value=512, max_value=1164, step=64),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("angular_activation", ["relu", "tanh"]))(x)
        #x = Dense(100, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("angular_units_2", min_value=50, max_value=200, step=32),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("angular_activation_2", ["relu", "tanh"]))(x)
        #x = Dense(50, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("angular_units_3", min_value=10, max_value=60, step=16),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("angular_activation_3", ["relu", "tanh"]))(x)
        #x = Dense(10, kernel_initializer="normal", activation="relu")(x)
        x = Dense(
            # Tune number of units
            units=hp.Int("angular_units_4", min_value=2, max_value=10, step=2),
            kernel_initializer="normal",
            # Tune the activation function to use.
            activation=hp.Choice("angular_activation_4", ["relu", "tanh"]))(x)
        x = Dense(1, kernel_initializer="normal", name="Angular")(x)

        return x

    @staticmethod
    def build(hp):
        width=200
        height=150
        input_shape = (height, width, 3)
        inputs = tf.keras.Input(shape=input_shape)
        linearVelocity = MyModel.build_linear_branch(hp, inputs)
        angularVelocity = MyModel.build_angular_branch(hp, inputs)

        model = tf.keras.Model(inputs=inputs, outputs=[linearVelocity, angularVelocity], name="MyModel")

        losses = {"Linear": "mse", "Angular": "mse"}
        lossWeights = {"Linear": 2, "Angular": 10}
        init_lr = 1e-3
        opt = tf.keras.optimizers.Adam(init_lr)
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics="mse")

        return model
