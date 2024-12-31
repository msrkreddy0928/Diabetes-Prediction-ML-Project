from sklearn.metrics import accuracy_score, classification_report,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    # probabilities_class_1 = model.predict_proba(X_test)[:, 1]
    # threshold = 0.3 
    # predicted_class = (probabilities_class_1 > threshold).astype(int)
    # print(X_test) 
    # print(predicted_class)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
   
    
    return accuracy, class_report

def training_accuracy_(model,X_train,Y_train):    
    Y_pred = model.predict(X_train)
    accuracy = accuracy_score(Y_train,Y_pred)

    return accuracy


def regg_evaluate_model(model,X_train,X_test):
    y_pred_train = model.predict(X_train)
    #print(y_pred_train)
    y_pred_prob_train = 1 / (1 + np.exp(-y_pred_train))
    y_pred_test = model.predict(X_test)
    y_pred_prob_test = 1 / (1 + np.exp(-y_pred_test))
    return y_pred_prob_train,y_pred_prob_test

def reg_evaluate(model,X_test,y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    return r2,mse


def FNN_evaluate(model,X_train,Y_train,X_test,y_test):
    eval = model.evaluate(X_test,y_test)

    return eval
    




def grid_search(model,param_grid,X_train,Y_train):
    grid_search = GridSearchCV(estimator= model, 
                           param_grid=param_grid, 
                           cv=5,
                           scoring='accuracy', )

    grid_search.fit(X_train,Y_train)
     
    return grid_search  
