# clear the terminal
clear

# run the data preprocess (temperature-single point)

# # 3 days to 1 hour
# python data_preprocess.py --file_path './.data/BelagaviTemperature.csv' --input_window 72 --output_window 1 --location_column ' Sulebhavi' --save_file_path './.data/processed/3D-to-1H_Temp_Sulebhavi.pkl'
# echo "Data preprocessed: ./.data/processed/3D-to-1H_Temp_Sulebhavi.pkl"

# # 1 day to 1 day
# python data_preprocess.py --file_path './.data/BelagaviTemperature.csv' --input_window 24  --output_window 24 --location_column ' Sulebhavi' --save_file_path './.data/processed/1D-to-1D_Temp_Sulebhavi.pkl'
# echo "Data preprocessed: ./.data/processed/1D-to-1D_Temp_Sulebhavi.pkl"

# # 1 week to 1 week
# python data_preprocess.py --file_path './.data/BelagaviTemperature.csv' --input_window 168 --output_window 168 --location_column ' Sulebhavi' --save_file_path './.data/processed/7D-to-7D_Temp_Sulebhavi.pkl'
# echo "Data preprocessed: ./.data/processed/7D-to-7D_Temp_Sulebhavi.pkl"

# # 1 day to 1 hour
# python data_preprocess.py --file_path './.data/BelagaviTemperature.csv' --input_window 24 --output_window 1 --location_column ' Sulebhavi' --save_file_path './.data/processed/1D-to-1H_Temp_Sulebhavi.pkl'
# echo "Data preprocessed: ./.data/processed/1D-to-1H_Temp_Sulebhavi.pkl"

# # 7 days to 1 hour
# python data_preprocess.py --file_path './.data/BelagaviTemperature.csv' --input_window 168 --output_window 1 --location_column ' Sulebhavi' --save_file_path './.data/processed/7D-to-1H_Temp_Sulebhavi.pkl'
# echo "Data preprocessed: ./.data/processed/7D-to-1H_Temp_Sulebhavi.pkl"

# 3 days to 7 days
python data_preprocess.py --file_path './.data/BelagaviTemperature.csv' --input_window 72 --output_window 168 --location_column ' Sulebhavi' --save_file_path './.data/processed/3D-to-7D_Temp_Sulebhavi.pkl'
echo "Data preprocessed: ./.data/processed/3D-to-7D_Temp_Sulebhavi.pkl"

# 1 day to 7 days
python data_preprocess.py --file_path './.data/BelagaviTemperature.csv' --input_window 24 --output_window 168 --location_column ' Sulebhavi' --save_file_path './.data/processed/1D-to-7D_Temp_Sulebhavi.pkl'
echo "Data preprocessed: ./.data/processed/1D-to-7D_Temp_Sulebhavi.pkl"

# 7 days to 7 days
python data_preprocess.py --file_path './.data/BelagaviTemperature.csv' --input_window 168 --output_window 168 --location_column ' Sulebhavi' --save_file_path './.data/processed/7D-to-7D_Temp_Sulebhavi.pkl'
echo "Data preprocessed: ./.data/processed/7D-to-7D_Temp_Sulebhavi.pkl"

# 1 day to 7 days
