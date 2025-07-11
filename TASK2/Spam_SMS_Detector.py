!pip install pandas scikit-learn gradio --quiet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import gradio as gr

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Create a pipeline: Vectorizer + Classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Prediction function
def classify_message(msg):
    prediction = model.predict([msg])[0]
    return "🚫 Spam" if prediction == 1 else "✅ Not Spam"

# Gradio UI
iface = gr.Interface(fn=classify_message,
                     inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
                     outputs="text",
                     title="📩 Spam Message Classifier",
                     description="Type a message and check if it's Spam or Not")

iface.launch()
