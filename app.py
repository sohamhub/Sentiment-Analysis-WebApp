from flask import Flask, render_template, request
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords

nltk.download('stopwords')

# Force correct template folder
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")

app = Flask(__name__, template_folder=template_dir)

# Load model
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

@app.route('/')
def home():
    return render_template("index.html", accuracy="78.25%")

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    prediction = prediction.lower().strip()

    return render_template(
        "index.html",
        prediction=prediction,
        accuracy="78.25%"
    )
    

    return render_template("index.html", prediction=prediction, accuracy="78.25%")

if __name__ == '__main__':
    app.run(debug=True)
