import pandas as pd

dataset_path = r'C:\Users\HP\Desktop\archive\dataset.csv'
data = pd.read_csv(dataset_path)

# Affiche les premières lignes
print(data.head())
