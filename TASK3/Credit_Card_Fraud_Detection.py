# 📦 Install required packages
!pip install pandas scikit-learn gradio openpyxl --quiet

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import gradio as gr

# 📁 Load the Excel file
df = pd.read_excel("Credit_card_Fraud_detection_dataset.csv")

# 🧹 Preprocess
# Encode categorical features: Location, DeviceType, IsInternational
le_location = LabelEncoder()
le_device = LabelEncoder()
le_international = LabelEncoder()

df['Location'] = le_location.fit_transform(df['Location'])
df['DeviceType'] = le_device.fit_transform(df['DeviceType'])
df['IsInternational'] = le_international.fit_transform(df['IsInternational'])

# 🎯 Features and Target
X = df[['Amount', 'Location', 'DeviceType', 'IsInternational']]
y = df['Class']

# 📊 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 🤖 Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 📈 Evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# 🌐 Gradio App
def predict_fraud(amount, location, device_type, is_international):
    # Encode inputs using the same encoders
    loc = le_location.transform([location])[0]
    dev = le_device.transform([device_type])[0]
    intl = le_international.transform([is_international])[0]

    x_input = [[amount, loc, dev, intl]]
    prob = model.predict_proba(x_input)[0][1]
    label = "FRAUD" if prob > 0.5 else "LEGIT"
    return f"Fraud Probability: {prob:.2f} → Prediction: {label}"

# Get dropdown options
location_options = list(le_location.classes_)
device_options = list(le_device.classes_)
intl_options = list(le_international.classes_)

# Gradio interface
gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Transaction Amount"),
        gr.Dropdown(location_options, label="Location"),
        gr.Dropdown(device_options, label="Device Type"),
        gr.Dropdown(intl_options, label="Is International")
    ],
    outputs="text",
    title="Credit Card Fraud Detection (Simple)",
    description="Enter transaction details to check if it's fraud or legit."
).launch()
