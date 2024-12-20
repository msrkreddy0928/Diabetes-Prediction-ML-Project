import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, chi2,f_classif,mutual_info_classif
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA


df =pd.read_csv("EDA/data/diabetes_prediction_dataset.csv")

# print(df.head())

# print(df.shape)         #(100000,9)

# print(df.isna().sum())  # No NUll values

# print(df.info())

# print(df.nunique())

# print(df.describe())

# print(df.duplicated().sum()) # 3854

df.drop_duplicates(inplace=True)

initial_df = df

# plt.figure(figsize=(15,8))
# sns.pairplot(initial_df.iloc[30000:],hue="diabetes",diag_kind='kde')
# plt.show()

# print(df.duplicated().sum())

# print(df.shape)  #(96146, 9)

# print(df['gender'].value_counts()) 

# print(df['diabetes'].value_counts())

df_majority = df[(df['diabetes']==0)]
# print("shape before removing from majority cat",df_majority.shape)  #(87664,9)

df_majority.drop(df_majority[df_majority['smoking_history'] == 'No Info'].index,inplace=True) 

# print("shape after removing from majority cat",df_majority.shape)    #(56222,9)

# for col in df_majority.columns:
#     print(len(df_majority[col]))

df_minority = df[(df['diabetes']==1)]

df = pd.concat([df_majority, df_minority])

""""
under_sampling = resample(df_majority,replace=True,n_samples=len(df_minority))

print(len(under_sampling))
# print(under_sampling)

# print(df['diabetes'].value_counts())

df = pd.concat([under_sampling, df_minority])

print(df['diabetes'].value_counts())
"""


df['smoking_history'] = df['smoking_history'].replace('ever','never')


df['age'] = df['age'].replace([1.16,1.88,1.08,1.40,1.72,1.32],1)
df['age'] = df['age'].replace(0.08,np.nan)

# print(df['age'].value_counts())

# print(df.shape)

categorical_features = []
continous_features = []
for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical_features.append(column)
    else:
        continous_features.append(column)
        
        
# print(continous_features)
# print(categorical_features)        
# plt.figure(figsize=(10, 8)) #(16964,9)

cleaned_df = df


continous_features_for_log = ['bmi','blood_glucose_level']

for col in continous_features:
    print(col,df[col].skew())

for col in continous_features_for_log:
    df[col] = np.log(df[col])
    
  
df.dropna(inplace=True)

# for col in continous_features:
#     print(col,df[col].skew())
    

continous_features_for_sqrt = ['HbA1c_level']

for col in continous_features_for_sqrt:
    df[col]=np.sqrt(df[col])


df_after_trans = df

for col in continous_features:
    print(col,df[col].skew())
    
# sns.pairplot(df_after_trans.iloc[30000:],hue="diabetes",diag_kind='kde')
# plt.show()    


Features_to_encode = ['gender','smoking_history']

""""

onehot = OneHotEncoder(sparse_output=False)

one_hot_encoder = onehot.fit_transform(df[Features_to_encode])

one_hot_df = pd.DataFrame(one_hot_encoder,columns=onehot.get_feature_names_out(Features_to_encode))

print(one_hot_df)

df = pd.concat([df,one_hot_df])

df = df.drop(Features_to_encode,axis=1)

print(df.head())
    
    """    
    
le = LabelEncoder()

for col in Features_to_encode:
    df[col] = le.fit_transform(df[col])
    

sns.pairplot(df.iloc[30000:],hue="diabetes",diag_kind='kde')
plt.show()    

    
x =df.drop(['diabetes'],axis=1)
y = df['diabetes']


selector = SelectKBest(mutual_info_classif,k=5)

x_new = selector.fit_transform(x,y)

selected_features = x.columns[selector.get_support()]

print("mutual_info",selected_features)


selector = SelectKBest(f_classif,k=5)

x_new = selector.fit_transform(x,y)

selected_features = x.columns[selector.get_support()]

print("f_classif",selected_features)


selector = SelectKBest(chi2,k=5)

x_new = selector.fit_transform(x,y)

selected_features = x.columns[selector.get_support()]


print("Chi2",selected_features)



model = RandomForestClassifier()
selector = RFE(model, n_features_to_select=5)
X_new = selector.fit_transform(x, y)
selected_features = x.columns[selector.support_]
print("Recursive Feature",selected_features)



model = RandomForestClassifier()
model.fit(x, y)

# Get feature importance
feature_importance = model.feature_importances_ 

print("Tree Based Models",feature_importance)


# print("PCA",importance)

# print(df.head())
# print(df['gender'].value_counts())
# print(df['smoking_history'].value_counts())

# i=1
# print(df.describe())


correlation = df.corr()

    






















    




  
    
    

    










