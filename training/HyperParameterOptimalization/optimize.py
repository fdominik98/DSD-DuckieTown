import argparse
import logging
import os
import time
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras_tuner

from model_for_optimalization import MyModel, NewModel
sys.path.insert(1, "../")
from log_reader import Reader # import from training/log_reader

MODEL_NAME = "MyModel"
logging.basicConfig(level=logging.INFO)


#! Default Configuration
EPOCHS = 10
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
        log_file,
        old_dataset,
        experimental,
        split,
    ):
        self.window = 4
        print("Observed TF Version: ", tf.__version__)
        print("Observed Numpy Version: ", np.__version__)

        #self.create_dir()

        try:
            self.observation, self.linear, self.angular = self.get_data(log_file, old_dataset)
        except Exception:
            try:
                self.observation, self.linear, self.angular = self.get_data(log_file, old_dataset)
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

        #model = self.configure_model(learning_rate=init_lr, epochs=epochs)
        #callbacks_list = self.configure_callbacks()

        tuner = keras_tuner.RandomSearch(
            hypermodel=NewModel.build, # The model-building function
            objective="val_loss", # The name of the objective to optimize (whether to minimize or maximize is automatically inferred for built-in metrics).
            max_trials=3, # The total number of trials to run during the search.
            executions_per_trial=3, # The number of models that should be built and fit for each trial. Different trials have different hyperparameter values.
            overwrite=True, # Control whether to overwrite the previous results in the same directory or resume the previous search instead.
            # directory="HyperParameterOptimalization", # A path to a directory for storing the search results.
            project_name="random_search_tuner", # The name of the sub-directory in the directory
        )
        tuner.search_space_summary() #  Print summary of the search space
        tuner.search(
            x=observation_train, y={"Linear": linear_train, "Angular": angular_train},
            validation_data=(
                observation_valid,
                {"Linear": linear_valid, "Angular": angular_valid},
                ),
            batch_size=64, epochs=10
            )
        # Get the top 2 models.
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]
        # Build the model.
        tuner.results_summary() # Print a summary of the search results
        best_model.summary()

        """# 11. GO!
        history = model.fit(
            x=observation_train,
            y={"Linear": linear_train, "Angular": angular_train},
            validation_data=(
                observation_valid,
                {"Linear": linear_valid, "Angular": angular_valid},
            ),
            epochs=EPOCHS,
            callbacks=callbacks_list,
            shuffle=True,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        model.save(f"trainedModel/{MODEL_NAME}.h5")"""

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
        model = MyModel.build(keras_tuner.HyperParameters())
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics="mse")
        return model

    def configure_callbacks(self):
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir="trainlogs/{}".format(f'{MODEL_NAME}-{datetime.now().strftime("%Y-%m-%d@%H-%M-%S")}')
        )

        filepath1 = f"trainedModel/{MODEL_NAME}Best_Validation.h5"
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(
            filepath1, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
        )

        # ? Keep track of the best loss model
        filepath2 = f"trainedModel/{MODEL_NAME}Best_Loss.h5"
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

        ########Data transformation###################
        observation = [ observation[x:x+self.window] for x in range(0,int(len(observation)/self.window)) ]
        print(1)
        linear = [ linear[x:x+self.window] for x in range(0,int(len(linear)/self.window)) ]
        print(2)
        angular = [ angular[x:x+self.window] for x in range(0,int(len(angular)/self.window)) ]
        print(3)

        print(np.array(observation).shape)
        print(np.array(linear).shape)
        print(np.array(angular).shape)
        ############# Datatransformation ###############

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
    parser.add_argument("--old_dataset", help="Set whether there is old datset", default=OLD_DATASET)
    parser.add_argument("--experimental", help="Set whether it is experimental", default=EXPERIMENTAL)
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
        old_dataset=args.old_dataset,
        experimental=args.experimental,
        split=float(args.split),
    )
