import os
import logging
from datetime import datetime
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from datetime import datetime
from data_preprocess import WeatherDataLoader, load_config
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import functools
import argparse


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"{func.__name__} completed in {elapsed_time:.2f} seconds")
        return result
    return wrapper

class SingleLocationWeatherPredictor:
    def __init__(self, config_path='config.yaml', **kwargs):
        self.config = load_config(config_path)
        self.config.update(kwargs)  # Override config with command-line arguments
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_experiment()
        self.setup_logging()
        self.load_data()
        self.build_model()

    def setup_experiment(self):
        self.experiment_name = f"Experiment_{self.timestamp}"
        
        # Extract filename from preprocessed_data_path
        self.data_filename = os.path.basename(self.config['preprocessed_data_path']).split('.')[0]
        
        # Setup paths for logs and plots
        self.log_dir = "./logs/week2"
        self.plot_dir = "./plots/week2"
        self.model_dir = self.config['model_save_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def get_timestamped_filename(self, base_name, extension):
        return f"{base_name}_{self.timestamp}.{extension}"

    def setup_logging(self):
        log_filename = os.path.join(self.log_dir, self.get_timestamped_filename(self.data_filename, "log"))
        logging.basicConfig(filename=log_filename, level=logging.INFO, 
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        
        logging.info(f"Logging started at: {self.timestamp}")
        logging.info(f"Experiment: {self.experiment_name}")
        logging.info(f"Preprocessed data path: {self.config['preprocessed_data_path']}")
        self.log_gpu_info()
        self.log_configs()

    def log_gpu_info(self):
        logging.info(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
        logging.info(f"Is GPU available: {tf.test.is_gpu_available()}")
        logging.info(f"Tensorflow built with CUDA: {tf.test.is_built_with_cuda()}")

    def log_configs(self):
        logging.info("Experiment Configurations:")
        config_to_log = {
            "batch_size": self.config['batch_size'],
            "train_split": self.config['train_split'],
            "shuffle": self.config['shuffle'],
            "model": {
                "lstm_units": self.config['model']['lstm_units'],
                "learning_rate": self.config['model']['learning_rate'],
                "epochs": self.config['model']['epochs']
            }
        }
        logging.info(json.dumps(config_to_log, indent=2))

    def load_data(self):
        self.data_loader = WeatherDataLoader.load(self.config['preprocessed_data_path'])
        self.train_dataset, self.test_dataset = self.data_loader()
        logging.info(f"Data loaded from {self.config['preprocessed_data_path']}")
        logging.info(f"Train dataset shape: {self.data_loader.train_shape}")
        logging.info(f"Test dataset shape: {self.data_loader.test_shape}")

    def build_model(self):
        self.model = Sequential([
            LSTM(self.config['model']['lstm_units'], activation='tanh', recurrent_activation='sigmoid', 
                 input_shape=(self.data_loader.input_window_size, 1)),
            Dense(self.data_loader.output_window_size)
        ])
        self.model.compile(optimizer=Adam(learning_rate=self.config['model']['learning_rate']), loss='mse')
        logging.info(f"Model built with {self.config['model']['lstm_units']} LSTM units")
        logging.info(f"Model summary:\n{self.model.summary()}")

    @timer
    def train(self):
        logging.info('Start training')
        checkpoint_filename = self.get_timestamped_filename(f'best_model_weights_{self.data_filename}', "h5")
        checkpoint_path = os.path.join(self.model_dir, checkpoint_filename)
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        self.history = self.model.fit(
            self.train_dataset, 
            epochs=self.config['model']['epochs'], 
            validation_data=self.test_dataset, 
            callbacks=[checkpoint],
            verbose=1
        )
        logging.info('Training completed')

    @timer
    def evaluate(self):
        logging.info('Start evaluation')
        test_loss = self.model.evaluate(self.test_dataset)
        logging.info(f"Test Loss: {test_loss}")

        predictions = self.model.predict(self.test_dataset)

        y_true = np.concatenate([y for x, y in self.test_dataset], axis=0)
        y_pred = predictions

        y_true_inv = self.data_loader.inverse_transform(y_true)
        y_pred_inv = self.data_loader.inverse_transform(y_pred)

        rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        r2 = r2_score(y_true_inv, y_pred_inv)

        logging.info(f"Root Mean Squared Error: {rmse}")
        logging.info(f"Mean Absolute Error: {mae}")
        logging.info(f"R-squared Score: {r2}")

        return y_true_inv, y_pred_inv

    def plot_results(self, y_true, y_pred):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[:100].flatten(), label='Actual')
        plt.plot(y_pred[:100].flatten(), label='Predicted')
        plt.legend()
        plt.title(f'Temperature Prediction - {self.data_filename}')
        plt.xlabel('Time')
        plt.ylabel('Temperature')

        plot_filename = self.get_timestamped_filename(f"{self.data_filename}_prediction", "png")
        plot_path = os.path.join(self.plot_dir, plot_filename)
        plt.savefig(plot_path)
        logging.info(f"Prediction plot saved as {plot_path}")

    def plot_history(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title(f'Model Training History - {self.data_filename}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        history_filename = self.get_timestamped_filename(f"{self.data_filename}_history", "png")
        history_path = os.path.join(self.plot_dir, history_filename)
        plt.savefig(history_path)
        logging.info(f"Training history plot saved as {history_path}")

    def run(self):

        self.train() # model will be saved in the model_save_dir automatically
        y_true, y_pred = self.evaluate()
        self.plot_results(y_true, y_pred)
        self.plot_history()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Location Weather Predictor")
    parser.add_argument("--preprocessed_data_path", required=True, type=str, help="Path to the preprocessed data file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")

    args = parser.parse_args()
    predictor = SingleLocationWeatherPredictor(**vars(args))
    predictor.run()