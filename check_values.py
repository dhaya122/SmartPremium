import pandas as pd

df = pd.read_csv("data/insurance.csv")

columns = [
    "Gender",
    "Marital Status",
    "Education Level",
    "Occupation",
    "Location",
    "Policy Type",
    "Customer Feedback",
    "Smoking Status",
    "Exercise Frequency",
    "Property Type"
]

for col in columns:
    print(f"\n{col}:")
    print(df[col].dropna().unique())