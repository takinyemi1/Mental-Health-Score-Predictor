# **Mental Health Score Predictor - Student Wellness Analytics**

This is an end-to-end machine learning web application that predicts a student's *mental health score* from user-provided features.
The project compares the models, *Linear Regression*, *Random Forest Regression*, and a *hyperparameter-tuned Random Forest model*, 
with a production-style preprocessing and inference workflow using *scikit-learn pipelines*, *ColumnTransformer*, *FastAPI*, and a web frontend using *HTML, CSS,* and *JavaScript*.

> **Important:** This project is intended for educational and informational purposes only. It is not a clinical diagnosis and does not
> simulate the advice from a qualified mental health professional.

## **Project Overview**

The purpose of this project is to build a reliable regression-based prediction system using common machine-learning practices.

This project includes:
- Linear Regression as the baseline model
- Random Forest Regression
- A Hyperparameter-tuned Random Forest using ```RandomizedSearchCV```
- ```Pipeline``` for combining preprocessing and model training
- ```ColumnTransformer``` to handle (e.g., scale, impute, encode) different feature types
- Train/test split before preprocessing to avoid data leakage
- Cross-validation for hyperparameter search
- Evaluation using R<sup>2</sup>, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)
- FastAPI REST API for model inference
- HTML, CSS, and JavaScript frontend

## **Machine Learning Workflow**

```text
User Input                         ->
Frontend (HTML/CSS/JavaScript)     ->
FastAPI -> Preprocessing Pipeline  ->
ColumnTransformer                  ->
Trained ML Model                   ->
Predicted Mental Health Score      ->
Frontend Response
```

### Live Demo

Try the deployed application here:

<a href="https://mental-health-score-predictor-xmla.onrender.com">Mental Health Score Predictor</a>

This application is deployed on Render and can be accessed directly from a web browser.

## **Models Used**

### **1. Linear Regression**

Used as a baseline regression model to establish a straightforward reference point for comparing model performance.

### **2. Random Forest Regression**

A tree-based ensemble model used to capture nonlinear relationships between input features and the targeted mental health score.

### **3. Hyperparameter-tuned Random Forest**

```RandomizedSearchCV``` with cross-validation was used to search through different Random Forest hyperparameter combinations and identify a suitable configuration.

## **Result**

It's important to note that the tuned model did not outperform the default Random Forest model. And though the margin was small, this further emphasizes why model selection should be 
based on validation/test performance rather than assumptions, because hyperparameter tuning won't always show better results.

## **Model Performance**

| Model | Test R<sup>2</sup> | Training R<sup>2</sup> | MAE | RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.740 | 0.724 | 0.536 | 0.676 |
| Random Forest | **0.878** | 0.981 | **0.347** | **0.463** |
| Random Forest (tuned) | 0.865 | 0.955 | 0.369 | 0.487 |

### **Evaluation Metrics**

- **R<sup>2</sup>:** Measures how much of the variation in mental health scores our model explains, from 0 (useless) to 1 (perfect). > 0.85 means the model explains 85% of the pattern.

- **MAE (Mean Absolute Error):** Measures, on average, how far off our prediction is (i.e., lower values indicate smaller average prediction errors).

- **RMSE (Root Mean Squared Error):** It's similar to MAE (i.e., lower values indicate better prediction performance), but penalizes big mistakes more heavily.

Based on the captured results, the **default Random Forest** achieved the strongest held-out test performance among the evaluated models, with a test R<sup>2</sup> of approximately **0.878**.

## **Tech Stack**

### **Machine Learning**
- Python
- Pandas
- NumPy
- Scikit-learn
- Linear Regression
- RandomizedSearchCV
- Cross-validation
- Pipeline
- ColumnTransformer

### **Frontend**
- HTML
- CSS
- JavaScript

### **Backend**
- FastAPI
- Uvicorn
- REST API

### **Project Structure**
```text
Mental-Health-Score-Predictor/
│
├── data/
│   └── Student_Social_Media_And_Mental_Health_Impact.csv
│
├── notebooks/
│   └── mental_health_score_predictor.ipynb
│
├── interface/
│   ├── assets/
│       └── mind.png
│   └── index.html
│   └── script.js
│   └── style.css
├── plots
│   └── avg_daily_use_hours_vs_mental_health_score_scatterplot.png
│   └── mental_health_score_heatmap.png
│   └── mental_health_score_hist.png
│   └── most_used_platform_countplot_ordered.png
│   └── most_used_platform_countplot.png
│   └── sleep_hours_per_night_vs_mental_health_score_scatter_plot.png
│   └── stress_level_vs_mental_health_score_boxplot.png
│
├── Mental_Health_Model.pkl
├── requirements.txt
├── README.md
├── main.py
└── ...
```
