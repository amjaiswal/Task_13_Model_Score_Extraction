# Task 13 - Model Score Extraction

## Objective

Develop a validated scoring interface for a machine learning model capable of predicting Titanic passenger survival using both single-record and batch inputs.

## Features

- Train and save a Random Forest model
- Load saved model using Joblib
- Single-record prediction
- Batch prediction
- Prediction probability
- Model versioning
- Input validation
- Error handling
- Save predictions to CSV

## Project Structure

```
Task_13_Model_Score_Extraction/
│
├── data/
├── models/
├── outputs/
├── predict.py
├── train_model.py
├── task13.py
├── requirements.txt
└── README.md
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the scoring interface:

```bash
python task13.py
```

## Output

- Single prediction
- Batch prediction
- Probability score
- Model version
- Input validation
- `outputs/predictions.csv`

## Dataset

Titanic Dataset