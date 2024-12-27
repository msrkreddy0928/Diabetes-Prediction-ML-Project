from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

model=joblib.load("best_model1.pkl")
xgboost = joblib.load('xgboost.pkl')
regg_model = joblib.load('regg_model.pkl')

# gender_encoder = joblib.load('gender_encoder.pkl')
smoking_history_encoder = joblib.load('smoking_history_encoder.pkl')
scaler = joblib.load('scaler.pkl')


@app.route("/")
def home():
    pred_text = None  

    return render_template('home1.html',prediction_text=pred_text)


@app.route('/Predict',methods=['POST'])
def predict():
    
    pred_text = None  
    if request.method == 'POST':
        try:
            # gender = request.form['gender']
            age  = request.form['age']
            # hypertension = request.form['hypertension']
            # heart_disease = request.form['heartdisease']
            smoking_history = request.form['smokinghistory']
            bmi=float(request.form['bmi'])
            HbA1c = float(request.form['HbA1c'])
            blood_glucose_level = float(request.form['bloodglucoselevel'])
          
        except:
            
            return render_template('home1.html', prediction_text="Invalid input. Please enter numeric values.")    
        
        # if hypertension=='yes':    
        #     hypertension=0
        # else:
        #     hypertension=1
            
        # if heart_disease == 'yes':
        #     heart_disease=1
        # else:
        #     heart_disease=0           
           
       
        # gender_encoded = gender_encoder.transform([gender])[0]
   
        smoking_history_encoded = smoking_history_encoder.transform([smoking_history])[0]
        
        bmi = np.log1p(bmi)
        blood_glucose_level = np.log1p(blood_glucose_level)

        input_data = [[age,smoking_history_encoded,bmi,HbA1c,blood_glucose_level]]
        
        scaled_input_data = scaler.transform(input_data)
        
        bg = xgboost.predict(scaled_input_data)
        
        scaled_input_data = pd.DataFrame(scaled_input_data)
        
        scaled_input_data[5] = bg
        
        y_pred = regg_model.predict(scaled_input_data)[0]
        print(y_pred)
        y_pred = np.round(y_pred*100,2)
     
      
         
    
        # prediction_prob = model.predict_proba(scaled_input_data)
        
        # print(prediction_prob)
        
        # if prediction_prob[0][0]>prediction_prob[0][1]:
        #     pred_tex = "Your diabetes results is negative with "+str(prediction_prob[0][0]*100)+"% accuracy"
        # else:
        #     pred_tex = "Your diabetes results is positive with "+str(prediction_prob[0][1]*100)+"% accuracy" 
        
        
        prediction = model.predict(scaled_input_data)
        
        prediction = (prediction[0] > 0.5).astype(int)
        
        print(prediction)

        if(prediction==0):
            if y_pred>49:
                pred_text = "Your diabetes results is negative and having "+str(y_pred)+"% of chances to prone to diabetes.As chances of diabetes is more please consult your doctor" 
            else:
                pred_text= "Your diabetes results is negative and having "+str(y_pred)+"% of chances to prone to diabetes." 
        else:
          pred_text= "Your diabetes results is positive and having "+str(np.round(100-y_pred,2))+"% of chances of not having diabetes.Please consult your doctor for further treatment"
          
        return render_template('home1.html', prediction_text=pred_text)
   
            

if __name__ == '__main__':  
    app.run(debug=True)
    
        
        

    
