from data_preprocess import WeatherDataLoader

loader = WeatherDataLoader().load('./.data/processed/Temp_Sulebhavi.pkl')

print(loader.X_train)
print("+"*50)
# test the scaler by inverse transforming the data
print(loader.inverse_transform(loader.X_train))
