from flask import Flask, render_template, request
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

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/getURL', methods=['POST'])
def getURL():
    if request.method == 'POST':
        url = request.form['url']
        print(f"🌐 Checking URL: {url}")

        # ✅ Extract Features
        data = FeatureExtraction.getAttributess(url)

        # ✅ Convert DataFrame to List (Fixes Shape Issues)
        if isinstance(data, pd.DataFrame):
            data = data.iloc[0].tolist()  # Convert first row to list
        elif isinstance(data, np.ndarray):
            data = data.flatten().tolist()  # Flatten NumPy array
        elif not isinstance(data, list):
            print("❌ ERROR: Feature extraction failed! Unexpected data type.")
            return render_template("home.html", error="Feature extraction failed!")

        # ✅ Validate Feature Count
        if len(data) != len(feature_names):
            print(f"❌ ERROR: Feature count mismatch! Expected {len(feature_names)}, got {len(data)}")
            return render_template("home.html", error="Feature extraction failed!")

        # ✅ Convert Data to DataFrame with Correct Feature Names
        df = pd.DataFrame([data], columns=feature_names)

        # ✅ Convert Categorical Columns to Numeric
        for col in df.columns:
            if df[col].dtype == "object":
                if col in encoders:
                    df[col] = encoders[col].transform(df[col].astype(str))
                else:
                    df[col] = df[col].astype("category").cat.codes  # Convert to category codes

        # ✅ Fill Missing Values
        df.fillna(0, inplace=True)

        # ✅ Convert to Float
        df = df.astype(np.float64)

        # ✅ Predict
        predicted_value = RFmodel.predict(df)
        result = "Legitimate" if predicted_value == 0 else "Phishing"

        return render_template("home.html", error=result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
