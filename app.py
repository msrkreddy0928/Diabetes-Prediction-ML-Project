from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from src.configuration import setup_logging




setup_logging()

# Initialize the Flask application
app = Flask(__name__)


# Loading pre-trained models,encoders,scalers
model=joblib.load("best_model1.pkl")
logging.info("random forest model is loaded")

xgboost = joblib.load('xgboost.pkl')
logging.info("xgboost model is loaded")

regg_model = joblib.load('regg_model.pkl')
logging.info("regression model is loaded")


# fnn_model = joblib.load('fnn_model.pkl')
smoking_history_encoder = joblib.load('smoking_history_encoder.pkl')
logging.info("smoking encoder is loaded")

scaler = joblib.load('scaler.pkl')
logging.info("scaler is loaded")


@app.route("/")
def home():
    
    """This route renders the homepage where users can input data to predict diabetes."""
    
    pred_text = None  
    return render_template('home1.html',prediction_text=pred_text)


@app.route('/Predict',methods=['POST'])
def predict():
    
    """
    This route processes the input data submitted by the user, applies necessary transformations,
    runs predictions using multiple models, and returns the prediction result to be displayed.
    """
    
    
    pred_text = None  
    if request.method == 'POST':
        try:
            age  = int(request.form['age'])
            smoking_history = request.form['smokinghistory']
            bmi=float(request.form['bmi'])
            HbA1c = float(request.form['HbA1c'])
            blood_glucose_level = float(request.form['bloodglucoselevel'])
            
            if age>110 or age<0:
                return render_template('home1.html', prediction_text="Invalid input age. Please enter valid age.")
            
            if bmi>100 or bmi<0:
                return   render_template('home1.html', prediction_text="Invalid input bmi. Please enter valid bmi value.")
            if HbA1c>18 or HbA1c<0:
                 return   render_template('home1.html', prediction_text="Invalid input HbA1c. Please enter valid hbA1c value.")
             
            if blood_glucose_level>300 or blood_glucose_level<0:
                 return   render_template('home1.html', prediction_text="Invalid input blood_glucose_level. Please enter valid blood_glucose_level value.")
             
            logging.info("input values are validated") 
          
        except:
            return render_template('home1.html', prediction_text="Invalid input. Please enter numeric values.")    
   
        smoking_history_encoded = smoking_history_encoder.transform([smoking_history])[0]
        logging.info("smoking values are enocded")
        
        
        
        bmi = np.log1p(bmi)
        blood_glucose_level = np.log1p(blood_glucose_level)
        logging.info("log transformations are applied for bmi and blood_glucose_level")
        

        input_data = [[age,smoking_history_encoded,bmi,HbA1c,blood_glucose_level]]
        
        scaled_input_data = scaler.transform(input_data)
        logging.info("Input values are scaled")
        
        bg = xgboost.predict(scaled_input_data)
        logging.info("predicted the values using xgbooster")
        
        scaled_input_data = pd.DataFrame(scaled_input_data)
        
        scaled_input_data[5] = bg
        
        y_pred = regg_model.predict(scaled_input_data)[0]
        logging.info("predicted the values using regression model")
        print(y_pred)   
        y_pred = np.round(y_pred*100,2)
        
        # pred = fnn_model.predict(scaled_input_data)
        # print("fnn prediction",pred)
        # prediction = (pred[0][0] > 0.5).astype(int)
        # print("fnn predict",prediction)
        
        # prediction_prob = model.predict_proba(scaled_input_data)
        
        # print(prediction_prob)
        
        # if prediction_prob[0][0]>prediction_prob[0][1]:
        #     pred_tex = "Your diabetes results is negative with "+str(prediction_prob[0][0]*100)+"% accuracy"
        # else:
        #     pred_tex = "Your diabetes results is positive with "+str(prediction_prob[0][1]*100)+"% accuracy" 
        
        
        pred_prob = model.predict_proba(scaled_input_data)
        print(pred_prob)  
        
        
        prediction = model.predict(scaled_input_data)
        logging.info("predicted the values using random forest model")
        
        prediction = (prediction[0] > 0.5).astype(int)
        
        print(prediction)
        
        if(prediction==0):
            if y_pred>49:
                pred_text = "Your diabetes results is negative and having "+str(y_pred)+"% of chances to prone to diabetes.As chances of diabetes is more please consult your doctor." 
            else:
                pred_text= "Your diabetes results is negative and having "+str(y_pred)+"% of chances to prone to diabetes." 
        else:
          pred_text= "Your diabetes results is positive and having "+str(np.round(100-y_pred,2))+"% of chances of not having diabetes.Please consult your doctor for further treatment."
        
        logging.info("output message is returned")  
        return render_template('home1.html', prediction_text=pred_text)
   
            
            
# Run the Flask app in debug mode for local testing
if __name__ == '__main__':  
    app.run(debug=True)
    
        
        

    
