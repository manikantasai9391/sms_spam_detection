# Data manipulation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Evaluation Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(df.head())
print(df.info())
print(df.tail())
print("Dataset Shape:", df.shape)

# =========================
# Check Missing Values
# =========================

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# Class Distribution
# =========================

print("\nClass Distribution:")
print(df["label"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="label", data=df)
plt.title("SMS Spam Distribution")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

# =========================
# Remove Duplicates
# =========================

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("New Dataset Shape:", df.shape)

# =========================
# Convert Labels
# =========================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# =========================
# Features & Target
# =========================

X = df["message"]
y = df["label"]

print("\nSample Messages:")
print(X.head())

print("\nSample Labels:")
print(y.head())

# =========================
# TF-IDF Vectorization
# =========================

tfidf = TfidfVectorizer(stop_words="english")

X_tfidf = tfidf.fit_transform(X)

print("\nTF-IDF Shape:", X_tfidf.shape)
print("First 20 Features:")
print(tfidf.get_feature_names_out()[:20])

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features:", X_train.shape)
print("Testing Features :", X_test.shape)

# =========================
# Train Model
# =========================

model = MultinomialNB()

model.fit(X_train, y_train)

# =========================
# Prediction
# =========================

y_pred = model.predict(X_test)

# =========================
# Accuracy
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)

# =========================
# Classification Report
# =========================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# =========================
# Train Model
# =========================

model = MultinomialNB()

model.fit(X_train, y_train)

# =========================
# Save Model & Vectorizer
# =========================

import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully!")
# =========================
# Predict User Input
# =========================

while True:
    message = input("\nEnter an SMS message (or type 'exit' to quit): ")

    if message.lower() == "exit":
        print("Program terminated.")
        break

    message_vector = tfidf.transform([message])

    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        print("Prediction: Spam")
    else:
        print("Prediction: Ham")

    probability = model.predict_proba(message_vector)

    print(f"Ham Probability : {probability[0][0]:.4f}")
    print(f"Spam Probability: {probability[0][1]:.4f}")