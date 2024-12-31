from sklearn.pipeline import Pipeline
import joblib
import tensorflow as tf
from keras import Sequential
from keras.layers import InputLayer,Dense
from keras.regularizers import l2



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

def FNN_train(X_train,y_train,X_test,y_test):
   model = Sequential()
   model.add(InputLayer(input_shape=(X_train.shape[1],)))
   model.add(Dense(64,activation='relu',kernel_initializer='he_normal',bias_initializer='ones'))
   model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
   model.add(Dense(1,activation='sigmoid'))
   model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
   model.fit(X_train,y_train,batch_size=64,epochs=10,validation_data=(X_test,y_test))
   FNN_save(model)
   
   return model
   
   
def FNN_save(model):
   joblib.dump (model,'fnn_model.pkl')
      
    

def save_model(model):
   joblib.dump(model,'best_model1.pkl')
   
def save_model_regg(model):
   joblib.dump(model,'regg_model.pkl')
      
   
       
    
    
    
  
    
    
    
    
    
    
    


 
