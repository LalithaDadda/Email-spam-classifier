import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Support common dataset column names
if {"v1", "v2"}.issubset(data.columns):
    data = data[["v1", "v2"]]
    data.columns = ["label", "message"]
elif {"label", "message"}.issubset(data.columns):
    data = data[["label", "message"]]
else:
    raise ValueError(
        "CSV must contain label/message or v1/v2 columns."
    )

# Clean the dataset
data = data.dropna()
data["label"] = data["label"].astype(str).str.strip().str.lower()
data["message"] = data["message"].astype(str)

data = data[data["label"].isin(["spam", "ham"])]

# Split input and output
X = data["message"]
y = data["label"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the AI/ML model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("Accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Predict a new message
print("\n--- Email Spam Classifier ---")
message = input("Enter an email or message: ")

prediction = model.predict([message])[0]

if prediction == "spam":
    print("Result: SPAM MESSAGE")
else:
    print("Result: HAM (NORMAL MESSAGE)")
