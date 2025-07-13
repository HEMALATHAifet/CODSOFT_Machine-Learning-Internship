# 💳 Credit Card Fraud Detection using Machine Learning

> 🎯 **Internship Task 3 - CodSoft Machine Learning Internship**  
> 🧠 **Day 9 – 100 Days AI Challenge**  
> 🔗 [My LinkedIn Post]()

---

## 📝 Problem Statement

The objective of this project is to **build a machine learning model** that can accurately detect whether a **credit card transaction is fraudulent or legitimate**. 

Credit card fraud is a major concern in today's digital world, and automated fraud detection systems can save companies and customers from significant financial losses.

---

## ✅ Project Highlights

- Built using **Logistic Regression**
- Frontend powered by **Gradio** for interactive prediction
- Lightweight and beginner-friendly

---

## 🌍 Why This Project Matters – Social Responsibility

This project is not just about accuracy or models. It’s about **protecting people**:

- Preventing **financial fraud** for ordinary people
- Helping banks and digital platforms build **trust**
- Promoting **AI for Good** – using machine learning to protect, not exploit

---

## 📊 Dataset Details

📁 File used: `Credit_card_Fraud_detection_dataset.csv`

### Columns Explained:

| Column Name       | Description |
|------------------|-------------|
| **Amount**        | The transaction amount in currency (₹ / $). High-value transactions are often targets for fraud. |
| **Location**      | Country where the transaction occurred. Certain countries may have a higher fraud risk. |
| **DeviceType**    | Device used for transaction – `Mobile` or `Desktop`. Mobile is usually more secure. |
| **IsInternational** | Whether the transaction was international – `Yes` or `No`. International transactions are riskier. |
| **Class**         | Target variable: `0` for Legit, `1` for Fraud. This is what we aim to predict.

These features were **carefully encoded and processed** to help the machine learning model understand them and make accurate predictions.

---

## 🧠 Solution – How I Built It

1. **Data Loading**  
   Used `pandas` to load and explore the Excel dataset.

2. **Preprocessing**  
   - Categorical columns were encoded using `LabelEncoder`
   - Features and labels were separated

3. **Model Training**  
   Trained a **Logistic Regression model** using `scikit-learn`  
   Split the data into training and testing sets for evaluation.

4. **Model Evaluation**  
   Used `classification_report` and `confusion_matrix` to verify model performance.

5. **Gradio Interface**  
   Built a simple UI using `Gradio` to allow users to:
   - Enter transaction details
   - Get fraud prediction
   - See **probability score**
   - View a **bar chart showing how each feature contributed**
---

## 🌟 What I Learned

* How to preprocess categorical data
* How to build a logistic regression model
* How to use Gradio for UI
* How to interpret model predictions using visualizations
* Importance of **explainable AI** for trust and transparency

---
