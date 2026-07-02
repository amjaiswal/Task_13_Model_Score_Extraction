import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/model.pkl")

# Required input columns
REQUIRED_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

# Model version
MODEL_VERSION = "v1.0"


def validate_input(data):
    """
    Validate input data before prediction.
    """

    # Check for empty input
    if data.empty:
        raise ValueError("Input data is empty.")

    # Check for missing columns
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def predict_single(data):
    """
    Predict survival for a single passenger.
    """

    df = pd.DataFrame([data])

    validate_input(df)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "meaning": "Likely Survived" if prediction == 1 else "Likely Did Not Survive",
        "model_version": MODEL_VERSION
    }


def predict_batch(df):
    """
    Predict survival for multiple passengers.
    """

    validate_input(df)

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    result = df.copy()

    result["Prediction"] = predictions
    result["Meaning"] = [
        "Likely Survived" if p == 1 else "Likely Did Not Survive"
        for p in predictions
    ]
    result["Probability"] = probabilities.round(4)
    result["Model_Version"] = MODEL_VERSION

    return result