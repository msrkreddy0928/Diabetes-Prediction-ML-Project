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
import joblib
import xgboost as xgb


def load_data(file_path):

    return pd.read_csv(file_path)


def clean_data(data):
    
    data.drop_duplicates(inplace=True)
    
    # data_majority = data[(data['diabetes']==0)]
    
    # data_majority.drop(data_majority[data_majority['smoking_history'] == 'No Info'].index,inplace=True)
    
    # data_minority = data[(data['diabetes']==1)]
 
    # data = pd.concat([data_majority, data_minority])
    
    data['smoking_history'] = data['smoking_history'].replace('ever','never')
    
    data['age'] = data['age'].replace([1.16,1.88,1.08,1.40,1.72,1.32],1)
    data['age'] = data['age'].replace(0.08,np.nan)
    
    categorical_features = []
    continous_features = []
    
    for column in data.columns:
        if len(data[column].unique()) <= 10:
            categorical_features.append(column)
        else:
          continous_features.append(column)
    

    # for col in continous_features:
    #   print(col,data[col].skew())        
    
    continous_features_for_log = ['bmi','blood_glucose_level']
    # for col in continous_features_for_log:
    #      data[col] = np.log(data[col])
    
    for col in continous_features_for_log:
      data[col] = np.log1p(data[col])
      # print(col,data[col].skew())
    
    data.dropna(inplace=True)
    
    
    return data

def preprocessed_data(data):
    
    
    data = clean_data(data)

    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns
    
   
    for col in categorical_features:

        label = LabelEncoder()
        label.fit(data[col])
        joblib.dump(label,col+"_encoder.pkl")
        data[col] = label.transform(data[col])

        
    # X = data.drop(['diabetes'],axis=1)
    
    # Y = data['diabetes']
    
    # perform_statistical_tests(X,Y)

    # selector = SelectKBest(mutual_info_classif,k=5)

    # x_new = selector.fit_transform(X,Y)

    # selected_features = X.columns[selector.get_support()]



    # selector = SelectKBest(f_classif,k=5)

    # x_new = selector.fit_transform(X,Y)

    # selected_features = X.columns[selector.get_support()]

    # # print("f_classif",selected_features)


    # selector = SelectKBest(chi2,k=5)

    # x_new = selector.fit_transform(X,Y)

    # selected_features = X.columns[selector.get_support()]


    # # print("Chi2",selected_features)

    # model = RandomForestClassifier()
    # selector = RFE(model, n_features_to_select=5)
    # X_new = selector.fit_transform(X,Y)
    # selected_features = X.columns[selector.support_]
    # # print("Recursive Feature",selected_features)

    # model = RandomForestClassifier()
    # model.fit(X, Y)

    # feature_importance = model.feature_importances_ 

    # # print("Tree Based Models",feature_importance)

    # After feature reduction the resulted features are


    X = data.loc[:,['age','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
    Y = data['diabetes']

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

    sc = StandardScaler()
     
    sc.fit(X_train)
    
    joblib.dump(sc,"scaler.pkl")
    
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)    
    
    scale_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])
   
    
    xgboost_model = xgb.XGBClassifier(scale_pos_weight=scale_weight ,random_state=42)
    xgboost_model.fit(X_train, Y_train)
    joblib.dump(xgboost_model,"xgboost.pkl")
    
    xgb_train_pred = xgboost_model.predict(X_train)
    xgb_test_pred = xgboost_model.predict(X_test)
    X_train_new = pd.DataFrame(X_train)
    X_test_new = pd.DataFrame(X_test)
    X_train_new[5] = xgb_train_pred
    X_test_new[5] = xgb_test_pred
    
    smote = SMOTE(random_state=42,k_neighbors=4)
    X_train_new,Y_train = smote.fit_resample(X_train_new,Y_train)

 
    return X_train_new,X_test_new,Y_train,Y_test



# def perform_statistical_tests(X,y):
   
#     numeric_features = ['age','bmi','HbA1c_level','blood_glucose_level']
#     categorical_features = ['gender','hypertension','heart_disease','smoking_history']
#     test_result = {}
    

#     group1 = X[numeric_features][y == y.unique()[0]]
#     group2 = X[numeric_features][y == y.unique()[1]]
    
#     t_stat, p_value = stats.ttest_ind(group1, group2)
#     print("Two-sample t-test:")
#     print(f"t-statistic: {t_stat}, p-value: {p_value}")
#     # if p_value < 0.05:
#     #     print("Reject the null hypothesis")
#     # else:
#     #     print("Fail to reject the null hypothesis")
#     # print()
  
#     for col in categorical_features:
#         print(col)
#         contingency_table = pd.crosstab(X[col], y)
#         chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
#         test_result[col] = {
#             'Chi-Square Statistic': chi2,
#             'p-value': p,
#             'Degrees of Freedom': dof,
#             'Dependent': p < 0.05  
#         }
#         print("Chi-Square Test Results for Categorical Variables:",col)
#         print(test_result)

