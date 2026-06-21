import pandas as pd

df = pd.read_csv("data/infrarisk_dataset.csv")

print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())