from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import pandas as pd
from transformers_intrusion import ProtoEncoding, ServiceEncoding, StateEncoding, ColumnDropper, FeatureScaler, Prediction
import json

app = Flask(__name__)

#Loding the IDS pipeline
pipeline = joblib.load('pipeline_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON input from the request form data
        json_input = request.form['jsonInput']
        input_data = json.loads(json_input) 

        #Ensuring that the input data is in dictionary format
        if not isinstance(input_data, dict):
            return jsonify({"error": "Input data should be a JSON object"}), 400

        #Convert input data to a DataFrame
        data = pd.DataFrame([input_data])

        #Ensure data is not empty
        if data.empty:
            return jsonify({"error": "Input data is empty"}), 400

        # Transform the data using the pipeline
        transformed_data = pipeline.transform(data)

        # Ensure transformed data is not empty
        if transformed_data.empty:
            return jsonify({"error": "Transformed data is empty"}), 400

        # Perform prediction using the model
        prediction = transformed_data['prediction'].values[0]

        # Redirect to the appropriate page based on prediction
        if prediction == 1:
            return jsonify(redirect = url_for('intrusion_detected'))
        else:
            return jsonify(redirect = url_for('normal_traffic'))

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