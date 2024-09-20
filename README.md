# Deep-Atmospheric-Predictor

This repository contains the code for the Deep Atmospheric Predictor, a machine learning model designed to predict atmospheric conditions such as temperature, humidity, and wind speed. The model is built using TensorFlow and Keras, and it uses long short-term memory (LSTM) networks to process time series data.

## Usage

```bash
# Preprocess the data
./preprocess.sh

# Train and Evaluate the SingleLocationWeatherPredictor model
./run_single.sh

# Train the SingleLocationWeatherPredictor model in the dataset within a floder
./run.sh
```