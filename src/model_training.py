from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib



def train_model(model,X_train, y_train):
   model.fit(X_train,y_train)
   return model
    

def save_model(model):
   joblib.dump(model,'best_model.pkl')
       
    
    
    
  
    
    
    
    
    
    
    


 