import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2,f_classif,mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score,classification_report
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging   



 # Global variables to store cleaned data, categorical features, continuous features, transformed data, and correlation matrix


cleaned_df = None
cat_features = None
con_features = None
trans_df = None
corr=None    

# load_data function: This function is used to load a CSV file and return a pandas DataFrame.

def load_data(file_path):

    return pd.read_csv(file_path)


# clean_data function: Cleans the data by removing duplicates, handling missing values,
# applying log transformation to certain columns, and categorizing features as categorical or continuous.

def clean_data(data):
    
    data.drop_duplicates(inplace=True)
    
     # Replacing specific 'ever' values to 'never' in the 'smoking_history' column
    data['smoking_history'] = data['smoking_history'].replace('ever','never')
    
    # Replacing some specific 'age' values and setting them to 1.
    data['age'] = data['age'].replace([1.16,1.88,1.08,1.40,1.72,1.32],1)
    data['age'] = data['age'].replace(0.08,np.nan)
    
    
    categorical_features = []
    continous_features = []
    
    for column in data.columns:
        if len(data[column].unique()) <= 10:
            categorical_features.append(column)
        else:
          continous_features.append(column)
    
    # Store global variables for further use
    global cleaned_df
    cleaned_df = data
    global cat_features 
    cat_features = categorical_features
    global con_features
    con_features = continous_features
    
    continous_features_for_log = ['bmi','blood_glucose_level']
    
    for col in continous_features_for_log:
      data[col] = np.log1p(data[col])
      
      
    # Drop rows with NaN values  
    data.dropna(inplace=True)
    
    global trans_df
    trans_df = data
    
    logging.info("data is cleaned")
    
    return data


# preprocessed_data function: Encodes categorical features, standardizes features,
# splits the data into training and testing sets, applies feature selection, 
# and trains an XGBoost model while handling class imbalance using SMOTE.
    
def preprocessed_data(data):
    
    
    data = clean_data(data)

    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns
    
     # Encode categorical features using LabelEncoder and save the encoders
    for col in categorical_features:

        label = LabelEncoder()
        label.fit(data[col])
        joblib.dump(label,col+"_encoder.pkl")
        data[col] = label.transform(data[col])

        
    X = data.drop(['diabetes'],axis=1)
    
    # Store correlation matrix for later use
    global corr
    corr = X.corr()
    Y = data['diabetes']
    
    # perform_statistical_tests(X,Y)
    
    #feature_selection(X,Y)
    
    

    # After feature reduction the resulted features are

    X = data.loc[:,['age','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
    Y = data['diabetes']
    
     # Split the data into training and testing sets (80-20 split)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
    
    logging.info("training and testing data is splitted in the ratio 80/20")
  
    
    # Standardize the values
    sc = StandardScaler()
     
    sc.fit(X_train)
    
    joblib.dump(sc,"scaler.pkl")
    
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
        
    
    scale_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])
   
      # Train the XGBoost model
    xgboost_model = xgb.XGBClassifier(scale_pos_weight=scale_weight ,random_state=42)
    xgboost_model.fit(X_train, Y_train)
    joblib.dump(xgboost_model,"xgboost.pkl")
    
     # Make predictions on both train and test data
    xgb_train_pred = xgboost_model.predict(X_train)
    xgb_test_pred = xgboost_model.predict(X_test)
    
    logging.info("XG Booster is applied")
    

    
    # Prepare new DataFrame and added the feature 
    X_train_new = pd.DataFrame(X_train)
    X_test_new = pd.DataFrame(X_test)
    X_train_new[5] = xgb_train_pred
    X_test_new[5] = xgb_test_pred
    
    # Used SMOTE for oversampling the minority class in the training set
    smote = SMOTE(random_state=42,k_neighbors=4)
    X_train_new,Y_train = smote.fit_resample(X_train_new,Y_train)
    
    logging.info("oversampling the minority class labels is done using smote")

 
    return X_train_new,X_test_new,Y_train,Y_test


# perform_statistical_tests function: Performs two-sample t-test for numeric features and 
# Chi-Square test for categorical features to check for feature significance.

def perform_statistical_tests(X,y):
   
    numeric_features = ['age','bmi','HbA1c_level','blood_glucose_level']
    categorical_features = ['gender','hypertension','heart_disease','smoking_history']
    test_result = {}
    

    group1 = X[numeric_features][y == y.unique()[0]]
    group2 = X[numeric_features][y == y.unique()[1]]
    
      # Perform two-sample t-test for numeric features
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print("Two-sample t-test:")
    print(f"t-statistic: {t_stat}, p-value: {p_value}")
    for col in categorical_features:
        print(col)
        contingency_table = pd.crosstab(X[col], y)
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        test_result[col] = {
            'Chi-Square Statistic': chi2,
            'p-value': p,
            'Degrees of Freedom': dof,
            'Dependent': p < 0.05  
        }
        print("Chi-Square Test Results for Categorical Variables:",col)
        print(test_result)
 
 
 # feature_selection function: Performs various feature selection techniques, including 
# SelectKBest (using mutual_info_classif, f_classif, and chi2), RFE with a Random Forest model,
# and tree-based model for feature importance evaluation.       
        
def feature_selection(X,Y):
    
    selector = SelectKBest(mutual_info_classif,k=5)

    x_new = selector.fit_transform(X,Y)

    selected_features = X.columns[selector.get_support()]

    print("Features selected from mutul_info_classif method",selected_features)

    selector = SelectKBest(f_classif,k=5)

    x_new = selector.fit_transform(X,Y)

    selected_features = X.columns[selector.get_support()]

    print("Features selected from f_classif method",selected_features)


    selector = SelectKBest(chi2,k=5)

    x_new = selector.fit_transform(X,Y)

    selected_features = X.columns[selector.get_support()]

    print("Features selected from Chi2 method",selected_features)

    model = RandomForestClassifier()
    selector = RFE(model, n_features_to_select=5)
    X_new = selector.fit_transform(X,Y)
    selected_features = X.columns[selector.support_]
    
    print("Features selected from Recursive method",selected_features)

    model = RandomForestClassifier()
    model.fit(X, Y)

    feature_importance = model.feature_importances_ 

    print("Features selected from Tree Based Models",feature_importance)

        

# Helper functions to return global variables related to the data and features

def get_cleaned_df():
    
    return cleaned_df

def get_cat_features():
    
    return cat_features

def get_con_features():
    
    return con_features

def get_trans_df():
    
    return trans_df

def get_corr():
    
    return corr