# ----------------------------
# FINAL CLEAN EDA VERSION
# ----------------------------

import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Users/soham/Downloads/archive (7)/twitter_airline_sentiment_clean.csv")

print("Columns are:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSentiment Count:")
print(df['airline_sentiment'].value_counts())

# ----------------------------
# Plot 1: Sentiment Count
# ----------------------------
plt.figure()
sns.countplot(x='airline_sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()

# ----------------------------
# SAFE Text Column Detection
# ----------------------------

if 'text' in df.columns:
    text_column = 'text'
elif 'clean_text' in df.columns:
    text_column = 'clean_text'
else:
    print("No text column found ❌")
    exit()

# ----------------------------
# Plot 2: Text Length
# ----------------------------

df['text_length'] = df[text_column].astype(str).apply(len)

plt.figure()
sns.histplot(df['text_length'], bins=50)
plt.title("Text Length Distribution")
plt.show()

# ----------------------------
# Plot 3: Text Length by Sentiment
# ----------------------------

plt.figure()
sns.boxplot(x='airline_sentiment', y='text_length', data=df)
plt.title("Text Length by Sentiment")
plt.show()

print("\nEDA Completed Successfully ✅")

# ==================================================
#                MODEL TRAINING
# ==================================================

import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# Detect text column safely
if 'text' in df.columns:
    text_column = 'text'
elif 'clean_text' in df.columns:
    text_column = 'clean_text'
else:
    print("No text column found ❌")
    exit()

# Text cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['processed_text'] = df[text_column].apply(clean_text)

# Features and Labels
X = df['processed_text']
y = df['airline_sentiment']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("\n================ MODEL RESULTS ================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save Model
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully ✅")
