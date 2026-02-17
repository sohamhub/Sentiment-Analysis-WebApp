# ✈️ Sentiment Analysis Web Application

An end-to-end Machine Learning project that classifies airline tweets into **Positive**, **Negative**, or **Neutral** sentiments and provides real-time predictions through a Flask web application.

---

## 📌 Project Overview

Social media generates millions of user opinions daily. This project automates sentiment detection using Machine Learning and deploys it through a web interface.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Clean and preprocess text data
- Convert text to numerical features using TF-IDF
- Train Machine Learning model
- Evaluate model performance
- Deploy real-time prediction web app

---

## 📊 Dataset

- Twitter Airline Sentiment Dataset
- Contains labeled airline tweets
- Classes:
  - Positive
  - Negative
  - Neutral

---

## 🧠 Machine Learning Pipeline

1. Text Cleaning
2. Stopword Removal
3. TF-IDF Vectorization
4. Logistic Regression Model
5. Class Balancing for Improved Neutral Detection

---

## 📈 Model Performance

- Algorithm: Logistic Regression
- Accuracy: ~78%
- Improved neutral prediction using `class_weight='balanced'`

---

## 🌐 Web Application

Built using:

- Python
- Flask
- HTML
- CSS

### Features:

- Real-time sentiment prediction
- Color-coded results:
  - 🟢 Green → Positive
  - 🔴 Red → Negative
  - 🟡 Yellow → Neutral
- Clean modern UI

---

## 🏗️ Project Structure
Sentiment-Analysis-WebApp/
│
├── app.py
├── model_train.py
├── sentiment_model.pkl
├── vectorizer.pkl
│
├── templates/
│ └── index.html
│
├── static/
│ └── style.css
│
└── README.md
