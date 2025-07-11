!pip install pandas scikit-learn --q
!pip install numpy
!pip install gradio --q
!pip install mlflow joblib --q
import pandas as pd
import numpy as np
import re

# Load dataset (you can upload your own .csv)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
df['message'] = df['message'].apply(lambda x: re.sub(r'\W+', ' ', x.lower()))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# MLflow logging
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spam-Detection-Colab")

with mlflow.start_run():
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
import gradio as gr

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_sms(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Spam 🚫" if pred == 1 else "Ham ✅"

gr.Interface(fn=predict_sms, inputs="text", outputs="text", title="📩 Spam SMS Detector").launch()
