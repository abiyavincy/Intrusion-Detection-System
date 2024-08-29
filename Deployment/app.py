from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import pandas as pd
from transformers import ProtoEncoding, ColumnDropper, FeatureScaler, Predictor
import json

app = Flask(__name__)
pipeline = joblib.load('pipeline_intrusion_det.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #Inputs to be passed
        input_data = {
            'dpkts': int(request.form['dpkts']),
            'sttl': int(request.form['sttl']),
            'smean': int(request.form['smean']),
            'ct_srv_src': int(request.form['ct_srv_src']),
            'proto': request.form['proto']
        }

        data = pd.DataFrame([input_data])
        transformed_data = pipeline.transform(data)

        #Redirecting to the appropriate page based on prediction
        if transformed_data == 1:
             return redirect(url_for('intrusion_detected'))
        else:
            return redirect(url_for('normal_traffic'))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/intrusion_detected')
def intrusion_detected():
    return render_template('intrusion_detected.html')

@app.route('/normal_traffic')
def normal_traffic():
    return render_template('normal_traffic.html')

if __name__ == '__main__':
    app.run(debug=True)