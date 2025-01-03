from sklearn.metrics import accuracy_score, classification_report,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import logging

# evaluate_model function: This function evaluates the model's performance on the test set.

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    logging.info("target values are predicted by random forest model on testing data")
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
   
    
    return accuracy, class_report

# training_accuracy_ function: This function calculates the model's accuracy on the training set.

def training_accuracy_(model,X_train,Y_train):    
    Y_pred = model.predict(X_train)
    accuracy = accuracy_score(Y_train,Y_pred)

    return accuracy

# regg_evaluate_model function: This function is used to predict the values from X_train and X_test and sigmoid function is applied on both the predictions.

def regg_evaluate_model(model,X_train,X_test):
    y_pred_train = model.predict(X_train)
    y_pred_prob_train = 1 / (1 + np.exp(-y_pred_train))
    y_pred_test = model.predict(X_test)
    y_pred_prob_test = 1 / (1 + np.exp(-y_pred_test))
    return y_pred_prob_train,y_pred_prob_test


# reg_evaluate function: This function evaluates regression models by calculating two common performance metrics:
# R-squared (r2_score) and Mean Squared Error (MSE)

def reg_evaluate(model,X_test,y_test):
    y_pred = model.predict(X_test)
    logging.info("target values are predicted by regression model on testing data")
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    return r2,mse


# evaluate_model function: This function evaluates the FNN model's performance on the test set.

# def FNN_evaluate(model,X_train,Y_train,X_test,y_test):
#     eval = model.evaluate(X_test,y_test)

#     return eval
    


# grid_search function: This function performs an exhaustive search over a specified parameter grid using cross-validation.
# It finds the best hyperparameters for the given model based on accuracy.

def grid_search(model,param_grid,X_train,Y_train):
    grid_search = GridSearchCV(estimator= model, 
                           param_grid=param_grid, 
                           cv=5,
                           scoring='accuracy', )

    grid_search.fit(X_train,Y_train)
     
    return grid_search  
