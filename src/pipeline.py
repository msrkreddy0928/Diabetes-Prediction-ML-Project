from  data_preprocessing import load_data,preprocessed_data
from model_training import train_model
from model_evaluation import evaluate_model
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


models = {  "Linear Regression": LinearRegression(),
                 "KNN" :KNeighborsClassifier(),
                 "SVM" : SVC(),
                 "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                }

accuracy_dict = {}

def run_pipeline(file_path):
    
    data = load_data(file_path)

    X_train,X_test,Y_train,Y_test = preprocessed_data(data)
    
    for i,model in models.items():
    
        model_trained = train_model(model,X_train,Y_train)
        
        accuracy, report = evaluate_model(model_trained, X_test,Y_test)
        
        accuracy_dict[i] = accuracy
        
        # print("score of ",i)
        # print(f"Model Accuracy: {accuracy * 100:.2f}%")
        # print(f"Classification Report:\n{report}")
        
    print("best model_Score",max(sorted(accuracy_dict.values())))
        
     

if __name__ == '__main__':
    run_pipeline('EDA/data/diabetes_prediction_dataset.csv') 