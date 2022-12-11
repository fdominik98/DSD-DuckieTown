import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from my_model import MyModel
from log_reader import Reader

MODEL_NAME = "MyModel"
logging.basicConfig(level=logging.INFO)


#! Default Configuration
EPOCHS = 10000
INIT_LR = 1e-3
BATCH_SIZE = 16
TRAIN_PERCENT = 0.8
LOG_FILE = "extended_dataset.log"

EXPERIMENTAL = False
OLD_DATASET = False

import os
from tensorflow.python.client import device_lib

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class DuckieTrainer:
    def __init__(
        self,
        epochs,
        init_lr,
        batch_size,
        log_dir,        
        model_name,
        log_file,
        split,
    ):
        self.model_name = model_name
        print("Observed TF Version: ", tf.__version__)
        print("Observed Numpy Version: ", np.__version__)

        self.create_dir()

        try:
            self.observation, self.linear, self.angular = self.get_data(log_file)
        except Exception:
            try:
                self.observation, self.linear, self.angular = self.get_data(log_file)
            except Exception:
                logging.error("Loading dataset failed... exiting...")
                exit(1)
        logging.info(f"Loading Datafile completed")

        # 2. Split training and testing
        (
            observation_train,
            observation_valid,
            linear_train,
            linear_valid,
            angular_train,
            angular_valid,
        ) = train_test_split(self.observation, self.linear, self.angular, test_size=1 - split, shuffle=True)

        model = self.configure_model(learning_rate=init_lr, epochs=epochs)

        callbacks_list = self.configure_callbacks()

        # 11. GO!
        history = model.fit(
            x=observation_train,
            y={"Linear": linear_train, "Angular": angular_train},
            validation_data=(
                observation_valid,
                {"Linear": linear_valid, "Angular": angular_valid},
            ),
            epochs=epochs,
            callbacks=callbacks_list,
            shuffle=True,
            batch_size=batch_size,
            verbose=0,
        )

        model.save(f"trainedModel/{self.model_name}.h5")

    def create_dir(self):
        try:
            os.makedirs("trainedModel")
        except FileExistsError:
            print("Directory already exists!")
        except OSError:
            print("Create folder for trained model failed. Please check system permissions.")
            exit()

    def configure_model(self, learning_rate, epochs):
        losses = {"Linear": "mse", "Angular": "mse"}
        lossWeights = {"Linear": 2, "Angular": 10}
        model = MyModel.build(200, 150)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics="mse")
        return model

    def configure_callbacks(self):
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="trainlogs/{}".format(f'{self.model_name}-{datetime.now().strftime("%Y-%m-%d@%H-%M-%S")}')
        )

        filepath1 = f"trainedModel/{self.model_name}Best_Validation.h5"
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
            filepath1, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )

        # ? Keep track of the best loss model
        filepath2 = f"trainedModel/{self.model_name}Best_Loss.h5"
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(
            filepath2, monitor="loss", verbose=1, save_best_only=True, mode="min"
        )

        return [checkpoint1, checkpoint2, tensorboard]

    def get_data(self, file_path, old_dataset=False):
        """
        Returns (observation: np.array, linear: np.array, angular: np.array)
        """
        reader = Reader(file_path)

        observation, linear, angular = reader.read() if old_dataset else reader.modern_read()

        logging.info(
            f"""Observation Length: {len(observation)}
            Linear Length: {len(linear)}
            Angular Length: {len(angular)}"""
        )
        return (np.array(observation), np.array(linear), np.array(angular))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameter Setup")
    parser.add_argument("--epochs", help="Set the total training epochs", default=EPOCHS)
    parser.add_argument("--learning_rate", help="Set the initial learning rate", default=INIT_LR)
    parser.add_argument("--batch_size", help="Set the batch size", default=BATCH_SIZE)
    parser.add_argument("--log_dir", help="Set the training log directory", default="")
    parser.add_argument("--log_file", help="Set the training log file name", default=LOG_FILE)
    parser.add_argument("--model_name", help="Set the training log file name", default=MODEL_NAME)
    parser.add_argument(
        "--split",
        help="Set the training and test split point (input the percentage of training)",
        default=TRAIN_PERCENT,
    )

    args = parser.parse_args()

    DuckieTrainer(
        epochs=int(args.epochs),
        init_lr=float(args.learning_rate),
        batch_size=int(args.batch_size),
        log_dir=args.log_dir,
        log_file=args.log_file,
        model_name = args.model_name,
        split=float(args.split),
    )
