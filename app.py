from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    transformed_news = vectorizer.transform([news])

    prediction = model.predict(transformed_news)[0]

    probability = model.predict_proba(transformed_news).max() * 100

    result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    return render_template(
        "index.html",
        prediction=result,
        confidence=f"{probability:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)