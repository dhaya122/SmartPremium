import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

def predict(input_data):
    df = pd.DataFrame([input_data])
    processed = preprocessor.transform(df)
    return model.predict(processed)[0]

if __name__ == "__main__":

    sample_input = {
        "id": 1,
        "Age": 35,
        "Gender": "Male",
        "Annual Income": 50000,
        "Marital Status": "Single",
        "Number of Dependents": 2,
        "Education Level": "Bachelor",
        "Occupation": "Engineer",
        "Health Score": 80,
        "Location": "Urban",
        "Policy Type": "Premium",
        "Previous Claims": 1,
        "Vehicle Age": 5,
        "Credit Score": 700,
        "Insurance Duration": 3,
        "Policy Start Date": "2023-01-01",
        "Customer Feedback": "Good",
        "Smoking Status": "No",
        "Exercise Frequency": "Weekly",
        "Property Type": "House"
    }

    result = predict(sample_input)

    print("Predicted Premium:", result)