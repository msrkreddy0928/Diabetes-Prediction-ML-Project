from sklearn.pipeline import Pipeline
import joblib
import logging
# import tensorflow as tf
# from keras import Sequential
# from keras.layers import InputLayer,Dense
# from keras.regularizers import l2



 #train_model function: This function trains the given model on the training data (X_train, y_train)
 #and saves the trained model using the `save_model` function.

def train_model(model,X_train, y_train):
   model.fit(X_train,y_train)
   logging.info("random forest model is trained")
   save_model(model)
   return model


def train_reg_model(model,X_train,y_train):
   model.fit(X_train,y_train)
   return model
   
 # regg_train function: This function trains a regression model and saves the model 
 # after training using the `save_model_regg` function.  

def regg_train(model,X_train,y_train):
   model.fit(X_train,y_train)
   logging.info("Regression model is trained")
   save_model_regg(model)
   return model

# def FNN_train(X_train,y_train,X_test,y_test):
#    model = Sequential()
#    model.add(InputLayer(input_shape=(X_train.shape[1],)))
#    model.add(Dense(64,activation='relu',kernel_initializer='he_normal',bias_initializer='ones'))
#    model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
#    model.add(Dense(1,activation='sigmoid'))
#    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#    model.fit(X_train,y_train,batch_size=64,epochs=10,validation_data=(X_test,y_test))
#    FNN_save(model)
   
#    return model
   

 # FNN_save function: This function saves the trained FeedForward Neural Network model to a file named 'fnn_model.pkl' using joblib.

def FNN_save(model):
   joblib.dump (model,'fnn_model.pkl')
   logging.info('FNN model is saved in fnn_model.pkl')
   
   
      
# save_model function: This function saves the trained machine learning 
# to a file named 'best_model1.pkl' using joblib. 
    
def save_model(model):
   joblib.dump(model,'best_model1.pkl')
   logging.info("random forest model is saved in best_model1.pkl")
   
   


 # save_model_regg function: This function saves the trained regression model to a file named 'regg_model.pkl' 
 # using joblib.  

def save_model_regg(model):
   joblib.dump(model,'regg_model.pkl')
   logging.info("Regression model is saved in regg_model.pkl")
      
   
       
    
    
    
  
    
    
    
    
    
    
    


 
