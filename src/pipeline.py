from  data_preprocessing import load_data,preprocessed_data,get_cleaned_df,get_cat_features,get_con_features,get_corr,get_trans_df,load_data_from_database
from model_training import train_model,save_model,train_reg_model,regg_train #FNN_train
from model_evaluation import evaluate_model,training_accuracy_,grid_search,regg_evaluate_model,reg_evaluate #FNN_evaluate
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC
import pandas as pd
from data_preprocessing import cleaned_df,cat_features,con_features,corr,trans_df
from Visualization import pair_plot,count_plot_data,data_distribution_after_trans,raw_data_distribution,plot_box_plots,plot_correlation_matrix
import joblib
import numpy as np
from configuration import setup_logging
import logging
import os
from dotenv import load_dotenv


setup_logging()

logger = logging.getLogger(__name__)


load_dotenv()

url = os.getenv('url')

models = { "Random Forest": RandomForestClassifier(class_weight='balanced',random_state=42)}



# Hyperparameter grids for model tuning

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



model_regg = LinearRegression()

accuracy_dict = {}

# Main function to run the pipeline, load data, preprocess it, train models, and evaluate performance

def run_pipeline(file_path):
    logging.info("pipeline started")
    
    # data = load_data(file_path)
    
    # logging.info("dataset loading completed")
    
    data = load_data_from_database(url)
    
    logging.info("dataset loading from database completed")
    
    
    X_train,X_test,Y_train,Y_test = preprocessed_data(data)
  

    cleaned_df = get_cleaned_df()
    cat_features = get_cat_features()
    con_features = get_con_features()
    trans_df = get_trans_df()
    corr = get_corr()
    

    # model_FNN = FNN_train(X_train,Y_train,X_test,Y_test)
        
    # eval = FNN_evaluate(model_FNN,X_train,Y_train,X_test,Y_test)
    
    # print("Using FNN")
    # print("loss:{:0.3f},accuracy:{:0.3f}".format(eval[0],eval[1]))
    
    for i,model in models.items():
        
        # if i =="Decision Tree":
        #     print("yes")
        #     grid_Se = grid_search(model,param_grid_DT,X_train,Y_train)
        # else:
        #     grid_Se = grid_search(model,param_grid_RF,X_train,X_test)    
            
        # print("best parameters for" +i+" "+grid_Se.best_params_)
        # print("Best Cross-Validation Score:"+i+grid_Se.best_score_)
    
    
        model_trained = train_model(model,X_train,Y_train)
        
        training_accuracy = training_accuracy_(model_trained,X_train,Y_train)
        
        
        accuracy,report = evaluate_model(model, X_test,Y_test)
       
        accuracy_dict[i] = accuracy

        print("score of ",i)
        print(f"training acuuracy:{training_accuracy*100:.2f}%")
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print(f"Classification Report:\n{report}")
        
   
     
    X_train_new = X_train.iloc[:,0:5]

    X_test_new = X_test.iloc[:,0:5]
 
    model_regg_trained = train_reg_model(model_regg,X_train_new,Y_train)

        
    y_pred_prob_train,y_pred_prob_test = regg_evaluate_model(model_regg_trained,X_train_new,X_test_new)
    
                                                                                                                                                                                                                                                                                                                                          
    
    model_regg_trained = regg_train(model_regg,X_train,y_pred_prob_train)
    
    r2,mse = reg_evaluate(model_regg_trained,X_test,y_pred_prob_test)
    
    print("r2 score is",r2)
    print("mean square error is",mse)
 
    print("best model_Score",max(sorted(accuracy_dict.values())))
    
    logging.info("pipeline is ended")





    #pair_plot(cleaned_df.iloc[0:50000],"diabetes")                           #pair plots
    #count_plot_data(cleaned_df,cat_features)                                 #count plots
    #raw_data_distribution(cleaned_df,con_features)                           #hist
    #data_distribution_after_trans(trans_df,con_features)                     #hist after transformations
    #plot_box_plots(cleaned_df,con_features)                                  #boxplots
    #plot_correlation_matrix(corr)                                            #heat map
    





# Entry point  - Run the pipeline with the given dataset

if __name__ == '__main__':
    run_pipeline('EDA/data/diabetes_prediction_dataset.csv') 
