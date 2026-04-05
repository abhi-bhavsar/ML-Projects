import os
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
application = app
base_dir = os.path.dirname(os.path.abspath(__file__))
ridge_model = pickle.load(open(os.path.join(base_dir, 'Models/ridge.pkl'), 'rb'))
standard_scaler = pickle.load(open(os.path.join(base_dir, 'Models/scaler.pkl'), 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        BUI = float(request.form.get('BUI'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC,DC,BUI, ISI, Classes, Region]])
        new_data_scaled = standard_scaler.transform(new_data)
        result = ridge_model.predict(new_data_scaled)

        return render_template('index.html', result=result[0])

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)