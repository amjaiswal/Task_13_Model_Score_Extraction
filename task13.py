import os
import pandas as pd

from predict import predict_single, predict_batch

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("        TASK 13 - MODEL SCORE EXTRACTION")
print("=" * 60)

# -----------------------------
# SINGLE RECORD PREDICTION
# -----------------------------
print("\nSingle Record Prediction\n")

sample = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

try:
    result = predict_single(sample)

    print("Prediction Result")
    print("-------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")

except Exception as e:
    print("Error:", e)

# -----------------------------
# BATCH PREDICTION
# -----------------------------
print("\n" + "=" * 60)
print("Batch Prediction")
print("=" * 60)

try:

    df = pd.read_csv("data/Titanic-Dataset.csv")

    batch = df[[
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]].head(10)

    predictions = predict_batch(batch)

    predictions.to_csv(
        "outputs/predictions.csv",
        index=False
    )

    print(predictions.head())

    print("\nPredictions saved successfully!")
    print("Location: outputs/predictions.csv")

except Exception as e:
    print("Error:", e)

# -----------------------------
# INVALID INPUT TEST
# -----------------------------
print("\n" + "=" * 60)
print("Input Validation Test")
print("=" * 60)

try:

    bad_input = pd.DataFrame({
        "Age": [22]
    })

    predict_batch(bad_input)

except Exception as e:

    print("Validation Successful!")
    print(e)