import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    # -------------------------
    # Date Feature Engineering
    # -------------------------
    df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"])

    df["Policy_Year"] = df["Policy Start Date"].dt.year
    df["Policy_Month"] = df["Policy Start Date"].dt.month
    df["Policy_Day"] = df["Policy Start Date"].dt.day

    df.drop("Policy Start Date", axis=1, inplace=True)

    # -------------------------
    # Target
    # -------------------------
    X = df.drop("Premium Amount", axis=1)
    y = df["Premium Amount"]

    # -------------------------
    # Column Types
    # -------------------------
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # -------------------------
    # Numerical Pipeline
    # -------------------------
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # -------------------------
    # Categorical Pipeline
    # -------------------------
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # -------------------------
    # Combined Preprocessor
    # -------------------------
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    # -------------------------
    # Train Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -------------------------
    # Transform Data
    # -------------------------
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":

    df = load_data("data/insurance.csv")

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Done preprocessing successfully!")