from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import FeatureExtraction
import pickle
import numpy as np
import pandas as pd

# ✅ Load Model & Encoders
with open("RandomForestModel.sav", "rb") as model_file:
    RFmodel = pickle.load(model_file)

with open("encoders.sav", "rb") as encoder_file:
    encoders = pickle.load(encoder_file)

# ✅ Extract Feature Names from Model Training
feature_names = RFmodel.feature_names_in_

app = Flask(__name__)
CORS(app)

# ✅ Routes
@app.route('/')
def index():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/how-it-works')
def how_it_works():
    return render_template("how-it-works.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/getURL', methods=['POST'])
def getURL():
    if request.method == 'POST':
        url = request.form['url']
        print(f"Checking URL: {url}")

        # ✅ Extract Features
        data = FeatureExtraction.getAttributess(url)

        # ✅ Convert DataFrame to List (Fixes Shape Issues)
        if isinstance(data, pd.DataFrame):
            data = data.iloc[0].tolist()
        elif isinstance(data, np.ndarray):
            data = data.flatten().tolist()
        elif not isinstance(data, list):
            print("❌ ERROR: Feature extraction failed! Unexpected data type.")
            return render_template("home.html", error="Feature extraction failed!")

        # ✅ Validate Feature Count
        if len(data) != len(feature_names):
            print(f"❌ ERROR: Feature count mismatch! Expected {len(feature_names)}, got {len(data)}")
            return render_template("home.html", error="Feature extraction failed!")

        # ✅ Convert Data to DataFrame
        df = pd.DataFrame([data], columns=feature_names)

        # ✅ Encode Categoricals
        for col in df.columns:
            if df[col].dtype == "object":
                if col in encoders:
                    df[col] = encoders[col].transform(df[col].astype(str))
                else:
                    df[col] = df[col].astype("category").cat.codes

        # ✅ Clean Data
        df.fillna(0, inplace=True)
        df = df.astype(np.float64)

        # ✅ Predict
        predicted_value = RFmodel.predict(df)
        result = "Legitimate" if predicted_value == 0 else "Phishing"

        return render_template("home.html", error=result)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'message': 'No URL provided', 'isPhishing': False, 'isSuspicious': False}), 400

    # ✅ Extract Features
    features = FeatureExtraction.getAttributess(url)
    if isinstance(features, pd.DataFrame):
        features = features.iloc[0].tolist()
    elif isinstance(features, np.ndarray):
        features = features.flatten().tolist()
    elif not isinstance(features, list):
        return jsonify({'message': 'Feature extraction failed!', 'isPhishing': False, 'isSuspicious': True}), 500

    if len(features) != len(feature_names):
        return jsonify({'message': 'Feature count mismatch!', 'isPhishing': False, 'isSuspicious': True}), 500

    features = [x.item() if hasattr(x, "item") else x for x in features]
    df = pd.DataFrame([features], columns=feature_names)

    for col in df.columns:
        if df[col].dtype == "object":
            if col in encoders:
                df[col] = encoders[col].transform(df[col].astype(str))
            else:
                df[col] = df[col].astype("category").cat.codes

    df.fillna(0, inplace=True)
    df = df.astype(np.float64)

    predicted_value = int(RFmodel.predict(df)[0])
    details = {k: (v.item() if hasattr(v, "item") else v) for k, v in zip(feature_names, features)}

    if predicted_value == 1:
        return jsonify({
            'message': 'Phishing website detected!',
            'isPhishing': True,
            'isSuspicious': False,
            'details': details
        })
    else:
        return jsonify({
            'message': 'This website appears safe.',
            'isPhishing': False,
            'isSuspicious': False,
            'details': details
        })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
