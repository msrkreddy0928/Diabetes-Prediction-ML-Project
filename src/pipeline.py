# from  data_preprocessing import load_data,preprocessed_data
# from model_training import train_model,save_model
# from model_evaluation import evaluate_model,training_accuracy_,grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import joblib

models = {  #"Logistic Regression": LogisticRegression(),
#                   "KNN" :KNeighborsClassifier(n_neighbors=4,p=1),
#                   "SVM" : SVC(kernel='poly'),
#                  "Decision Tree": DecisionTreeClassifier(criterion='entropy',max_depth=30,splitter='best'),
                 "Random Forest": RandomForestClassifier(class_weight='balanced',random_state=42) }

param_grid_DT ={
    'criterion': ['gini', 'entropy'],        
    'max_depth': [None, 10, 20, 30, 40,],     
    'max_features': ['sqrt', 'log2'], 
    'splitter': ['best', 'random'],           
  
}


param_grid_RF = {
    'n_estimators': [100, 200, 300, 500],  
    'criterion': ['gini', 'entropy'],        
    'max_depth': [None, 10, 20, 30, 40, 50],  
    'min_samples_split': [2, 10, 20],        
    'min_samples_leaf': [1, 5, 10],            
    'max_features': ['sqrt', 'log2'], 
    'bootstrap': [True, False],         
       
                            
}

accuracy_dict = {}

def run_pipeline(file_path):
  
  data = load_data(file_path)
  
  X_train,X_test,Y_train,Y_test = preprocessed_data(data)
  
  for i,model in models.items():
        
        # if i =="Decision Tree":
        #     print("yes")
        #     grid_Se = grid_search(model,param_grid_DT,X_train,Y_train)
        # else:
        #     grid_Se = grid_search(model,param_grid_RF,X_train,X_test)    
            
        # print("best parameters for" +i+" "+grid_Se.best_params_)
        # print("Best Cross-Validation Score:"+i+grid_Se.best_score_)
    #     new_data = {
    # 'age': [73],
    # 'smoking_history': ['former'],
    # 'bmi': [25.91],
    # 'HbA1c_level': [9],
    # 'blood_glucose_level': [160]
    #  }


        # new_data_df = pd.DataFrame(new_data)


   
        # smoking_encoder = joblib.load('smoking_history_encoder.pkl')

        # new_data_df['bmi'] = np.log1p(new_data_df['bmi'])
        # new_data_df['blood_glucose_level']=np.log1p(new_data_df['blood_glucose_level'])  


        # new_data_df['smoking_history'] = smoking_encoder.transform(new_data_df['smoking_history'])
        
        # sc = joblib.load('scaler.pkl')

        # new_data_df  = sc.transform(new_data_df)
        
        # xgboost = joblib.load('xgboost.pkl')

        # xgb_test_pred = xgboost.predict(new_data_df)

        # new_data_df = pd.DataFrame(new_data_df)

        # new_data_df[5] = xgb_test_pred
    
        model_trained = train_model(model,X_train,Y_train)
        
        # pred = model_trained.predict(new_data_df)
        # print(pred)
        
        training_accuracy = training_accuracy_(model_trained,X_train,Y_train)
        
        
        accuracy,report = evaluate_model(model, X_test,Y_test)
       
        accuracy_dict[i] = accuracy

        print("score of ",i)
        print(f"training acuuracy:{training_accuracy*100:.2f}%")
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print(f"Classification Report:\n{report}")
        
    #  print("best model_Score",max(sorted(accuracy_dict.values())))


        
 

if __name__ == '__main__':
    run_pipeline('/content/diabetes_prediction_dataset.csv') 
