# Diabetes Prediction Web Application with Machine Learning

## Overview
This is a web application for predicting diabetes based on various health parameters using machine learning algorithms. The application uses a dataset to train different models and deploys a Flask-based web app for user interaction. The project involves data gathering, preprocessing, model training, evaluation, and deployment.

## Table of Contents
1. [Data Gathering](#data-gathering)
2. [Descriptive Analysis](#descriptive-analysis)
3. [Data Visualizations](#data-visualizations)
4. [Data Preprocessing](#data-preprocessing)
5. [Data Modelling](#data-modelling)
6. [Model Evaluation](#model-evaluation)
7. [Web Application](#web-application)
8. [Model Deployment](#model-deployment)
9. [Installation Instructions](#installation-instructions)
10. [Usage](#usage)
11. [Technologies](#technologies)


## Data Gathering
- **Source**: The dataset is collected from Kaggle.
- **Dataset Link**: [Link to Dataset](https://www.kaggle.com/dataset-link)
- **Number of Samples**: 100,001
- **Number of Features**: 9

**Features in the Dataset**:
- **Gender**
- **Age**
- **Hypertension**
- **Heart Disease**
- **Smoking History**
- **BMI**
- **HbA1c Level**
- **Blood Glucose Level**
- **Diabetes (Target Variable)**

## Descriptive Analysis
- Analyzed the mean, median, and standard deviation using `df.describe()`.
- Checked for null values using `df.isnull()` and duplicate values using `df.duplicated()`. 
- **3854 duplicate values were found and removed.**
- After removing duplicates, the dataset shape became (96146, 9).

### Class Imbalance:
- **0 (No Diabetes)**: 87,664 samples
- **1 (Diabetes Positive)**: 8,482 samples

To balance the dataset:
- Removed rows where "smoking history" was "No Info."
- After removing these rows, the dataset shape became (64,709, 9) with class distribution of:
  - **0 (No Diabetes)**: 56,222 samples
  - **1 (Diabetes Positive)**: 8,482 samples

### Feature Modification:
- Combined 'ever' and 'never' categories in the "smoking history" feature.
- Applied log and square root transformations to continuous features (such as BMI and blood glucose level) to reduce skewness.

## Data Visualizations
- **Pair plot** was plotted to show relationships between features.
- **Bar plots** were used for categorical features.
- **Histograms** were plotted before and after transformations to visualize the data distribution.
- **Heatmap** was created to show correlations between features.

## Data Preprocessing
- Categorical features were encoded using `LabelEncoder`.
- Correlations between features and the target variable were identified, and the most impactful features for predicting diabetes were:
  - **Age**
  - **Smoking History**
  - **HbA1c Level**
  - **Blood Glucose Level**
- Removed less impactful features (e.g., Gender, Hypertension, Heart Disease) for efficiency.
- Split the dataset into training and testing sets (80/20).
- Scaled the data using `StandardScaler`, and the scaler was saved using `joblib`.
- used **Xgboost** algorithm to add weights column to the datset as it is imbalnced.
- Used **SMOTE** to oversample the minority class during training.

## Data Modelling
- Used the following machine learning algorithms:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
- Trained models on the preprocessed data and calculated accuracy scores.

## Model Evaluation
- Evaluated model performance using classification metrics such as:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Each model was tested on the test data, and the classification report was generated to assess model performance.

## Web Application
- The web application was created using **Flask**.
- The trained model, encoder, and scaler and xgboost were loaded using `joblib.load()`.
- A **POST** method was used to handle user input from an HTML form.
- The form includes 5 fields to take input from the user for prediction.
- The user input is encoded, scaled,added weights using xgboost and passed through the trained model for prediction.
- **Predicted Output**:
  - If the model predicts "1," the output is: "The diabetes test is positive."
  - If the model predicts "0," the output is: "The diabetes test is negative."

## Model Deployment
- The model was deployed on **render**.
- Created an account on render, connected the GitHub repository, and deployed the application.
- Link to the deployed application: [Your App Link](https://yourapp.vercel.app)

## Installation Instructions
 Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   
##Navigate to the project directory:
cd diabetes-prediction

##Install dependencies:
bash
Run the Flask application:
python newapp.py

Usage
Navigate to the application URL (e.g., http://localhost:5000).
Enter the required details in the input form ( age, smoking_history,BMI, blood_glucose_level).
Click the "Predict" button to get the result.

Technologies
Python 3.10
Flask for the web framework
Html and css for form creation and styling
scikit-learn for machine learning models
Pandas for data manipulation
Matplotlib, Seaborn for data visualization
XGBoost for adding weights feature
SMOTE for oversampling the minority class
joblib for saving and loading the model, encoder, and scaler
render for deployment




