import joblib
import mlflow
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_preprocessing import (
    load_data,
    preprocess_data
)

print("Loading dataset...")

df = load_data("data/insurance.csv")

# Use 20,000 rows for faster training
df = df.sample(n=20000, random_state=42)

print("Dataset shape:", df.shape)

X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

print("Starting XGBoost training...")

mlflow.set_experiment("SmartPremium")

with mlflow.start_run():

    model = XGBRegressor(
        n_estimators=20,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Training completed!")

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    rmse = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    r2 = r2_score(
        y_test,
        preds
    )

    print("\nRESULTS")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    joblib.dump(
        model,
        "models/model.pkl"
    )

    joblib.dump(
        preprocessor,
        "models/preprocessor.pkl"
    )

    print("\nModel saved!")
    print("Preprocessor saved!")