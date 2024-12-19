from flask import Flask,render_template,request,redirect,url_for
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


newapp = Flask(__name__)

model = joblib.load('best_model.pkl')

gender_encoder = joblib.load('gender_encoder.pkl')
smoking_history_encoder = joblib.load('smoking_history_encoder.pkl')
scaler = joblib.load('scaler.pkl')


@newapp.route("/")
def home():
    pred_text = None  

    return render_template('home1.html',prediction_text=pred_text)


@newapp.route('/Predict',methods=['POST'])
def predict():
    
    pred_text = None  
    if request.method == 'POST':
        try:
            gender = request.form['gender']
            age  = request.form['age']
            hypertension = request.form['hypertension']
            heart_disease = request.form['heartdisease']
            smoking_history = request.form['smokinghistory']
            bmi=request.form['bmi']
            HbA1c = request.form['HbA1c']
            blood_glucose_level = request.form['bloodglucoselevel']
          
        except:
            
            return render_template('home1.html', prediction_text="Invalid input. Please enter numeric values.")    
        
        if hypertension=='yes':    
            hypertension=0
        else:
            hypertension=1
            
        if heart_disease == 'yes':
            heart_disease=1
        else:
            heart_disease=0           
           
       
        gender_encoded = gender_encoder.transform([gender])[0]
   
        smoking_history_encoded = smoking_history_encoder.transform([smoking_history])[0]

        input_data = [[gender_encoded,age,hypertension,heart_disease,smoking_history_encoded,bmi,HbA1c,blood_glucose_level]]
        
        scaled_input_data = scaler.transform(input_data)
    
        prediction = model.predict(scaled_input_data)
        
        prediction = (prediction[0] > 0.5).astype(int)
       
        if(prediction==0):
            pred_text="Your Diabetes result is negative"
        else:
          pred_text="Your Diabetes result is positive"
          
        return render_template('home', prediction_text=pred_text)
   
        
        
    

if __name__ == '__main__':  
    newapp.run(debug=True)
    

