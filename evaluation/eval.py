import argparse

import tensorflow as tf
import logging
import numpy as np
import sys

sys.path.insert(1, "../training")
from log_reader import Reader


class DuckieEvaluator:
    def __init__(
        self,
        batch_size,
        log_file,
        model_path
    ):
        try:
            self.observation, self.linear, self.angular = self.get_data(log_file)
        except Exception:
            logging.exception("message")
            logging.error("Loading dataset failed... exiting...")
            exit(1)
        print(f"Loading Datafile completed")
        print(f"Loading model")
        try:
            self.model = tf.keras.models.load_model(model_path)
        except:
            print("Couldn\'t load model")
            exit(1)

        print("Evaluate on test data")
        results = self.model.evaluate(self.observation, y={"Linear": self.linear, "Angular": self.angular}, batch_size= batch_size)

    def get_data(self, file_path, old_dataset=False):
        """
        Returns (observation: np.array, linear: np.array, angular: np.array)
        """
        reader = Reader(file_path)

        observation, linear, angular = reader.read() if old_dataset else reader.modern_read()

        print(
            f"""Observation Length: {len(observation)}
            Observation Shape: {observation[0].shape}
            Linear Length: {len(linear)}
            Angular Length: {len(angular)}"""
        )
        return (np.array(observation), np.array(linear), np.array(angular))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Parameter Setup")

    parser.add_argument("--batch_size", help="Set the batch size", default=128)
    parser.add_argument("--log_file", help="Set the training log file name", default="test_dataset.log")
    parser.add_argument("--model_path", help="Set the training log file name", default="MyModelBest_Validation.h5")

    args = parser.parse_args()

    DuckieEvaluator(
        batch_size=int(args.batch_size),
        log_file=args.log_file,
        model_path=args.model_path
    )


