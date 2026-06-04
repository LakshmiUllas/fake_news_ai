import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading dataset...")

df = pd.read_csv("data/news.csv")

df = df.dropna()

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

pickle.dump(
    model,
    open("models/fake_news_model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("models/vectorizer.pkl", "wb")
)

print("Model saved successfully!")