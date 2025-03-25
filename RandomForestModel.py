import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ✅ Load Dataset
legitimate_urls = pd.read_csv(r"C:\Aditya\Study\Phishing-Website-Detection-master\extracted_csv_files\legitimate-urls.csv")
phishing_urls = pd.read_csv(r"C:\Aditya\Study\Phishing-Website-Detection-master\extracted_csv_files\phishing-urls.csv")

# ✅ Merge Data
urls = pd.concat([legitimate_urls, phishing_urls], ignore_index=True)

# ✅ Drop Unnecessary Columns (Modify If Needed)
urls = urls.drop(columns=["Unnamed: 0"], errors="ignore")

# ✅ Separate Features & Labels
X = urls.drop(columns=["label"])  # Features
y = urls["label"]  # Target (0 = Legitimate, 1 = Phishing)

# ✅ Handle Categorical Columns
encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col])

# ✅ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ Train Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Save Model & Encoders
pickle.dump(rf_model, open(r"C:\Aditya\Study\Phishing-Website-Detection-master\Phishing website detection using UI\RandomForestModel.sav", "wb"))
pickle.dump(encoders, open(r"C:\Aditya\Study\Phishing-Website-Detection-master\Phishing website detection using UI\encoders.sav", "wb"))

print("✅ Model & Encoders saved successfully!")
