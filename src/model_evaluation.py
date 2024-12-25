from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


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




def grid_search(model,param_grid,X_train,Y_train):
    grid_search = GridSearchCV(estimator= model, 
                           param_grid=param_grid, 
                           cv=5,
                           scoring='accuracy', )
    print("yes")
    grid_search.fit(X_train,Y_train)
     
    return grid_search  
