import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template

application = Flask(__name__)
app = application;

linear_model = pickle.load(open('models/california_housing_model.pkl', 'rb'))
Standard_scalar = pickle.load(open('models/california_housing_scalar.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')    

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        MedInc = float(request.form['MedInc'])
        HouseAge = float(request.form['HouseAge'])
        AveRooms = float(request.form['AveRooms'])
        AveBedrms = float(request.form['AveBedrms'])
        Population = float(request.form['Population'])
        AveOccup = float(request.form['AveOccup'])
        Latitude = float(request.form['Latitude'])
        Longitude = float(request.form['Longitude'])
        
        data = [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]]
        scaled_data = Standard_scalar.transform(data)
        result = linear_model.predict(scaled_data)
        
        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')
        
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)