import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle
import os
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)['default']

CONFIG = load_config()

class WeatherDataLoader:
    def __init__(self, args=None):
        self.file_path = args.file_path if args else CONFIG['file_path']
        self.location_column = args.location_column if args else CONFIG['location_column']
        self.input_window_size = args.input_window if args else CONFIG['input_window_size']
        self.output_window_size = args.output_window if args else CONFIG['output_window_size']
        self.batch_size = args.batch_size if args else CONFIG['batch_size']
        self.train_split = args.train_split if args else CONFIG['train_split']
        self.shuffle = args.shuffle if args else CONFIG['shuffle']

        if args and args.save_file_path:
            self.save_file_path = args.save_file_path
        else:
            # Construct the save file path
            base_filename = os.path.basename(self.file_path).split('.')[0]
            save_dir = os.path.join(CONFIG['save_file_path'], base_filename)
            self.save_file_path = os.path.join(save_dir, f'{self.location_column}_weather_data_loader.pkl')
        
        self.scaler = MinMaxScaler()
        if args:
            self.load_and_preprocess_data()
            self.create_tf_datasets()

    def load_and_preprocess_data(self):
        # Load the data, skipping the first three rows
        df = pd.read_csv(self.file_path)
        
        # Select only the specified feature column
        if self.location_column not in df.columns:
            raise ValueError(f"Column '{self.location_column}' not found in the CSV file.")
        
        df = df[[self.location_column]]  # This ensures we have a DataFrame, not a Series
        df = df.iloc[3:] # skip the location_name row, latitude and longitude rows

        # Split data into train and test sets
        train_size = int(len(df) * self.train_split)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        
        # Fit the scaler on training data and transform both train and test
        self.scaler.fit(train_data)
        scaled_train_data = self.scaler.transform(train_data)
        scaled_test_data = self.scaler.transform(test_data)
        
        # Create sequences
        X_train, y_train = self.create_sequences(scaled_train_data)
        X_test, y_test = self.create_sequences(scaled_test_data)
        
        # Reshape input for LSTM [samples, time steps, features]
        self.X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        self.y_train = y_train
        self.y_test = y_test

    def create_sequences(self, data):
        X, y = [], []
        data = data.flatten()  # Flatten the 2D array to 1D
        for i in range(len(data) - self.input_window_size - self.output_window_size + 1):
            X.append(data[i:(i + self.input_window_size)])
            y.append(data[(i + self.input_window_size):(i + self.input_window_size + self.output_window_size)])
        return np.array(X), np.array(y)

    def create_tf_datasets(self):
        # Create TensorFlow datasets
        self._train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        if self.shuffle:
            self._train_dataset = self._train_dataset.shuffle(buffer_size=1000)
        self._train_dataset = self._train_dataset.batch(self.batch_size)

        self._test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        self._test_dataset = self._test_dataset.batch(self.batch_size)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1))

    def save(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.save_file_path), exist_ok=True)

        with open(self.save_file_path, 'wb') as f:
            pickle.dump({
                'file_path': self.file_path,
                'save_file_path': self.save_file_path,
                'scaler': self.scaler,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'X_test': self.X_test,
                'y_test': self.y_test,
                'location_column': self.location_column,
                'input_window_size': self.input_window_size,
                'output_window_size': self.output_window_size,
                'batch_size': self.batch_size,
                'train_split': self.train_split,
                'shuffle': self.shuffle
            }, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        loader = cls()  # Create instance without args
        
        # Set attributes from loaded data
        loader.file_path = data.get('file_path', None)
        loader.save_file_path = data.get('save_file_path', os.path.dirname(filename))
        loader.location_column = data['location_column']
        loader.input_window_size = data['input_window_size']
        loader.output_window_size = data['output_window_size']
        loader.batch_size = data['batch_size']
        loader.train_split = data['train_split']
        loader.shuffle = data['shuffle']
        loader.scaler = data['scaler']
        loader.X_train = data['X_train']
        loader.y_train = data['y_train']
        loader.X_test = data['X_test']
        loader.y_test = data['y_test']
        
        loader.create_tf_datasets()
        
        return loader

    def __call__(self):
        return self.train_dataset, self.test_dataset

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def train_shape(self):
        return self.X_train.shape, self.y_train.shape

    @property
    def test_shape(self):
        return self.X_test.shape, self.y_test.shape

    def __str__(self):
        return f"""
                {'='*50}\n
                WeatherDataLoader configuration:\n
                {'-'*50}\n
                File path:          {self.file_path}\n
                Save file path:     {self.save_file_path}\n
                Location column:    {self.location_column}\n
                Input window size:  {self.input_window_size}\n
                Output window size: {self.output_window_size}\n
                Batch size:         {self.batch_size}\n
                Train split:        {self.train_split:.2f}\n
                Shuffle:            {self.shuffle}\n
                Train shape:        {self.train_shape}\n
                Test shape:         {self.test_shape}\n
                Scaler:            {self.scaler}\n
                {'='*50}
            """

if __name__ == "__main__":
    import argparse

    # Test the WeatherDataLoader
    parser = argparse.ArgumentParser(description="Weather Prediction Model")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--save_file_path", type=str, help="Path to save processed data and model")
    parser.add_argument("--location_column", type=str, default=' Sulebhavi', help="Location column name")
    parser.add_argument("--input_window", type=int, default=24*7, help="Input window size")
    parser.add_argument("--output_window", type=int, default=1, help="Output window size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the data")
    
    args = parser.parse_args()

    loader = WeatherDataLoader(args)

    # print(loader)

    # # Test __call__
    # train_dataset, test_dataset = loader()
    # print("Train dataset:", train_dataset)
    # print("Test dataset:", test_dataset)

    # # Iterate over the train dataset
    # print("Iterating over train dataset:")
    # for i, (X, y) in enumerate(train_dataset):
    #     print(f"Batch {i + 1}:")
    #     print("X shape:", X.shape)
    #     print("y shape:", y.shape)
    #     if i == 2:  # Print only first 3 batches
    #         break

    # Test save and load
    loader.save()
    # loaded_loader = WeatherDataLoader.load(os.path.join(loader.save_file_path, 'weather_data_loader.pkl'))
    # print("Loaded loader:")
    # print(loaded_loader)
