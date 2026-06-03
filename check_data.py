import pandas as pd

df = pd.read_csv("data/insurance.csv")

print("Shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget statistics:")
print(df["Premium Amount"].describe())