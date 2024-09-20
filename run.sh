#!/bin/bash

# Clear the terminal
clear

# Specify the directory containing the .pkl files
DATA_DIR="./.data/processed"

# Loop through all .pkl files in the specified directory
for pkl_file in "$DATA_DIR"/*.pkl; do
    if [ -f "$pkl_file" ]; then
        echo "Building model for file: $pkl_file"
        
        # Run the Python script with the current .pkl file
        python SingleLocationWeatherPredictor.py --preprocessed_data_path "$pkl_file"
        
        # Check the exit status of the Python script
        if [ $? -ne 0 ]; then
            echo "Error occurred while building model for $pkl_file"
            echo "Exiting the script."
            exit 1
        fi
        
        echo "Finished building and evaluating model for: $pkl_file"
        echo "------------------------"
    fi
done

echo "All models built successfully."