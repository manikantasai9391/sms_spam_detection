from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "models", "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "models", "tfidfvectorizer.pkl"), "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    if prediction == 1:
        result = "🚨 Spam Message"
        confidence = round(probability[1] * 100, 2)
    else:
        result = "✅ Ham Message"
        confidence = round(probability[0] * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
