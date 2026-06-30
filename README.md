# Telco Customer Churn Prediction

## Live Application

https://churn-analyser-euarz19l7-partha-shs-projects.vercel.app/

## Overview

This project is an end-to-end Machine Learning application designed to predict customer churn in the telecommunications industry. The model analyzes customer demographics, account information, subscription details, and service usage patterns to identify customers who are likely to leave the service.

The project covers the complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, API development, and deployment.

## Dataset

The project uses the Telco Customer Churn dataset from Kaggle, which contains customer information such as:

* Demographic details
* Account information
* Service subscriptions
* Billing details
* Contract types
* Churn status

Dataset Source: Telco Customer Churn Dataset (Kaggle)

## Features

### Data Preprocessing

* Missing value handling
* Data type conversion
* One-Hot Encoding for categorical features
* Feature scaling using StandardScaler
* Automated preprocessing with ColumnTransformer

### Feature Engineering

* Average Charges Per Month
* Long-Term Customer Indicator
* High Monthly Charges Indicator
* Total Services Count

### Machine Learning.    

* Logistic Regression
* Random Forest Classifier
* Class imbalance handling
* Pipeline-based workflow
* Feature importance analysis

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Joblib
* Uvicorn
* Vercel

## Project Structure

```text
Churn-Analyser/
│
├── app.py
├── churn_model.pkl
├── requirements.txt
├── notebook.ipynb
├── README.md
└── vercel.json
```

## Model Pipeline

1. Data Cleaning
2. Feature Engineering
3. Train-Test Split
4. Feature Scaling and Encoding
5. Model Training
6. Model Evaluation
7. Model Serialization
8. API Deployment

## API Endpoint

### Predict Customer Churn

**Endpoint**

```http
POST /predict
```

**Input**

Customer information including demographics, subscription details, and billing information.

**Output**

```json
{
    "prediction": "Churn"
}
```

or

```json
{
    "prediction": "No Churn"
}
```

## Deployment

The application is deployed using Vercel and serves real-time churn predictions through a FastAPI backend.

Live URL:

https://churn-analyser-euarz19l7-partha-shs-projects.vercel.app/

## Future Improvements

* Hyperparameter tuning
* Cross-validation analysis
* XGBoost implementation
* SHAP explainability
* Docker containerization
* User authentication
* Advanced analytics dashboard

## Author

Partha Sharma
