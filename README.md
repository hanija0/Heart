# Heart
# Cardiovascular Risk Prediction

## Overview
This project aims to predict the **10-Year Cardiovascular Disease (CHD) Risk** using machine learning techniques. The dataset includes medical and lifestyle attributes such as age, blood pressure, cholesterol levels, smoking habits, and more. A **Random Forest Classifier** is used for prediction after preprocessing and handling class imbalance.

## Dataset
The dataset is stored in `data_cardiovascular_risk.csv` and contains:
- **Features:** Age, sex, smoking status, cholesterol levels, blood pressure, BMI, glucose, and other medical attributes.
- **Target:** `TenYearCHD` (Binary: 1 = High risk, 0 = Low risk)

## Preprocessing Steps
1. **Load Dataset**: The dataset is loaded using Pandas.
2. **Handle Missing Values**:
   - Categorical values are encoded (e.g., `sex`, `is_smoking`).
   - Numerical missing values are imputed using the mean strategy.
3. **Encode Categorical Variables**:
   - Label Encoding for `sex` and `is_smoking`.
4. **Class Imbalance Handling**:
   - **SMOTE (Synthetic Minority Over-sampling Technique)** is used to balance the dataset.
5. **Train-Test Split**:
   - Data is split into **80% training** and **20% testing**.

## Model Training
- **Algorithm Used:** `RandomForestClassifier`
- **Train the Model:** The model is trained on `X_train` and `y_train`.
- **Predict on Test Data:** The model predicts `y_test`.

## Evaluation Metrics
- **Accuracy:** `90.5%`
- **Precision, Recall, F1-score:** Computed for both classes (0 = No Risk, 1 = High Risk)

### Classification Report
```
              precision    recall  f1-score   support

           0       0.87      0.96      0.91       589
           1       0.96      0.85      0.90       563

    accuracy                           0.91      1152
   macro avg       0.91      0.90      0.90      1152
weighted avg       0.91      0.91      0.90      1152
```

## How to Run
1. **Mount Google Drive in Colab:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Load the Dataset:**
   ```python
   import pandas as pd
   df = pd.read_csv('/content/drive/MyDrive/data_cardiovascular_risk.csv')
   ```
3. **Run the Code Cells:** Execute each cell to preprocess data, train the model, and evaluate it.
The model is deployed on Raspberry Pi. The circuit diagram is provided in circuit.jpg.


## Dependencies
- Python
- Pandas
- Scikit-learn
- Imbalanced-learn (for SMOTE)
- Google Colab (if running in the cloud)

## Future Improvements
- Try advanced models like **XGBoost or Neural Networks**.
- Tune hyperparameters for better performance.
- Use feature engineering to improve predictions.

## Conclusion
This project effectively predicts **10-year cardiovascular disease risk** using a **Random Forest model** with **SMOTE** for handling class imbalance. The accuracy of **90.5%** shows promising results.

