# Task 13 - Model Score Extraction

## Overview

This project demonstrates a complete Machine Learning Model Scoring Interface using the Titanic dataset. It trains a Random Forest Classifier, saves the trained model, and provides a validated prediction interface for both single-record and batch scoring with probability scores, model versioning, and input validation.

---

## Objective

Develop a validated scoring interface that:

- Supports single-record prediction
- Supports batch prediction
- Returns prediction probability
- Includes model version information
- Validates input data
- Handles errors gracefully
- Saves prediction results for downstream use

---

## Dataset

**Dataset:** Titanic Dataset

**Target Variable:** `Survived`

---

## Features

- Train Random Forest Classifier
- Save trained model using Joblib
- Load saved model for inference
- Single-record prediction
- Batch prediction
- Probability score extraction
- Prediction meaning
- Model versioning
- Input validation
- Error handling
- Export batch predictions to CSV

---

## Project Structure

```
Task_13_Model_Score_Extraction/
│
├── data/
│   └── Titanic-Dataset.csv
│
├── models/
│   └── model.pkl
│
├── outputs/
│   └── predictions.csv
│
├── train_model.py
├── predict.py
├── task13.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn

---

## Workflow

1. Load Titanic dataset
2. Preprocess the data
3. Train Random Forest model
4. Save trained model
5. Load model for inference
6. Validate input data
7. Predict single record
8. Predict batch records
9. Return probability score and model version
10. Save batch predictions to CSV

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

### Run the Scoring Interface

```bash
python task13.py
```

---

## Sample Output

### Single Prediction

```
Prediction : 0
Probability : 0.11
Meaning : Likely Did Not Survive
Model Version : v1.0
```

### Batch Prediction

```
Prediction
Probability
Meaning
Model_Version
```

Batch prediction results are automatically saved to:

```
outputs/predictions.csv
```

---

## Input Validation

The scoring interface validates:

- Required feature columns
- Empty input
- Missing columns

Invalid inputs return meaningful error messages.

---

## Model Information

- Model: Random Forest Classifier
- Serialization: Joblib
- Model Version: v1.0

---

## Key Learning Outcomes

- Machine Learning model serialization
- Production-ready prediction interface
- Batch and real-time inference
- Input validation
- Model versioning
- Probability score extraction
- Error handling

---

## Author

**Amar Jaiswal**
