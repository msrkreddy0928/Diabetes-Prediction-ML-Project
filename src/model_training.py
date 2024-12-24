
from sklearn.pipeline import Pipeline
import joblib



def train_model(model,X_train, y_train):
   model.fit(X_train,y_train)
   save_model(model)
   return model
    

def save_model(model):
   joblib.dump(model,'best_model1.pkl')
   


   
   
       
    
    
    
  
    
    
    
    
    
    
    


 