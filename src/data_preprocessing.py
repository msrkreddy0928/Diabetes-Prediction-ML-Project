import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import joblib

def load_data(file_path):

    return pd.read_csv(file_path)


def clean_data(data):
    
    data.drop_duplicates(inplace=True)
    
    data_majority = data[(data['diabetes']==0)]
    
    data_majority.drop(data_majority[data_majority['smoking_history'] == 'No Info'].index,inplace=True)
    
    data_minority = data[(data['diabetes']==1)]

    data = pd.concat([data_majority, data_minority])
    
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
            
    
    continous_features_for_log = ['bmi','blood_glucose_level']

    for col in continous_features_for_log:
        data[col] = np.log(data[col])
        
    continous_features_for_sqrt = ['HbA1c_level']

    for col in continous_features_for_sqrt:
        data[col]=np.sqrt(data[col])    
    
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
  

        
    X = data.drop(['diabetes'],axis=1)
    
    Y = data['diabetes']
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
     
    sc.fit(X_train)
    
    joblib.dump(sc,"scaler.pkl")
    
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)    
    
    smote = SMOTE(random_state=22,k_neighbors=4)
    
    # undersample = RandomUnderSampler(random_state=42)

    X_train,Y_train = smote.fit_resample(X_train,Y_train)
 
    return X_train,X_test,Y_train,Y_test

