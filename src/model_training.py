from sklearn.pipeline import Pipeline
import joblib



def train_model(model,X_train, y_train):
   model.fit(X_train,y_train)
   save_model(model)
   return model



def train_reg_model(model,X_train,y_train):
   model.fit(X_train,y_train)
   return model
   
   
def regg_train(model,X_train,y_train):
   model.fit(X_train,y_train)
   save_model_regg(model)
   return model
   
   
    

def save_model(model):
   joblib.dump(model,'best_model1.pkl')
   
def save_model_regg(model):
   joblib.dump(model,'regg_model.pkl')
      
   
       
    
    
    
  
    
    
    
    
    
    
    


 
