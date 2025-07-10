!pip install pandas scikit-learn --q

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Step 1: Clean text
def clean_text(text):
    return re.sub(r'\W+', ' ', text.lower())

# Step 2: Load and preprocess training data
train_data = []
with open("train_data.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ::: ")
        if len(parts) == 4:
            _, _, genre, desc = parts
            genre_list = [g.strip() for g in genre.split(",")]
            train_data.append((desc.strip(), genre_list))

df = pd.DataFrame(train_data, columns=["description", "genres"])
df["clean_desc"] = df["description"].apply(clean_text)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_desc"])

# Step 4: Multi-label Genre Binarization
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["genres"])

# Step 5: Train the Model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X, y)

# Step 6: Get user input
print("\n🎬 Enter a new movie to predict genre:")
title = input("Enter Movie Title: ")
year = input("Enter Release Year: ")
description = input("Enter Movie Description: ")

# Step 7: Preprocess & Predict
cleaned = clean_text(description)
vec = vectorizer.transform([cleaned])

# Get prediction probabilities
proba = model.predict_proba(vec)[0]

# Use a probability threshold to allow low-confidence matches
threshold = 0.2
pred = [1 if p >= threshold else 0 for p in proba]
predicted_genres = [genre for genre, flag in zip(mlb.classes_, pred) if flag]

# Step 8: Show result
print(f"\n📽️ Movie: {title} ({year})")
print(f"📝 Description: {description}")
if predicted_genres:
    print(f"🎯 Predicted Genre(s): {', '.join(predicted_genres)}")
else:
    print("🎯 Predicted Genre(s): Unknown")

# Optional: Show all genre probabilities for debugging
print("\n🔍 Prediction Probabilities:")
for genre, p in zip(mlb.classes_, proba):
    print(f"{genre:15s}: {p:.3f}")
