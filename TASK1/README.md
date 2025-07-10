# 🎬 Movie Genre Prediction using TF-IDF and Logistic Regression

This project predicts the **genre(s)** of a movie based on its **description** using **machine learning**.

## 📁 Dataset Files

Place the following files in the project root or Colab workspace:

- `train_data.txt`: Training data in the format  
  `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
- `test_data.txt`: Test data for predictions  
  `ID ::: TITLE ::: DESCRIPTION`
- `test_data_solution.txt`: Correct genres for test set (used for evaluation)
- `description.txt`: Overview file (optional)

---

## 🧠 Model

- **Text Vectorization**: `TfidfVectorizer`
- **Classifier**: `LogisticRegression` wrapped in `OneVsRestClassifier` for **multi-label classification**
- **Evaluation**: `classification_report` and prediction confidence scores

---

## 🚀 How to Use

### ▶️ 1. Install Required Packages

```bash
pip install pandas scikit-learn
````

> If using Google Colab, these are preinstalled.

---

### ▶️ 2. Run the Model Training

```python
# Load and preprocess train_data.txt
# Train the model using TF-IDF + Logistic Regression
# Code provided in main.py or Colab notebook
```

---

### ▶️ 3. Predict Genre for New Movie

After training the model, it will prompt:

```bash
Enter Movie Title: MadhaGajaRaja
Enter Release Year: 2025
Enter Movie Description: A young man decides to help out two of his childhood friends...
```

And return:

```
🎯 Predicted Genre(s): action, drama
```

---

### ▶️ 4. Example Output

```
📽️ Movie: MadhaGajaRaja (2025)
📝 Description: A young man decides to help out two of his childhood friends...
🎯 Predicted Genre(s): drama, thriller

🔍 Prediction Probabilities:
drama          : 0.85
thriller       : 0.41
comedy         : 0.05
...
```

---

## ✅ Features

* Multi-label genre classification
* Custom genre prediction for user input
* Threshold-based prediction for low-confidence genres
* Outputs genre probabilities for transparency

---

## 📚 Future Improvements

* Add support for deep learning models (e.g., BERT)
* Build a Gradio or Streamlit web interface
* Save and load the model for reuse
* Use ensemble models for improved accuracy

---
