import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("financial_fraud_simulated_dataset.csv")
X = df.drop(columns=["fraud"])
y = df["fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

encoder = OrdinalEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

model = CategoricalNB(alpha=1.0)
model.fit(X_train_encoded, y_train)
y_pred = model.predict(X_test_encoded)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="Yes"))
print("Recall:", recall_score(y_test, y_pred, pos_label="Yes"))
print("F1-score:", f1_score(y_test, y_pred, pos_label="Yes"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=["No", "Yes"]))
print("Classification Report:\n", classification_report(y_test, y_pred))

example = pd.DataFrame([{
    "amount_level": "High",
    "country_risk": "HighRisk",
    "device_type": "Unknown",
    "login_time": "Unusual",
    "merchant_type": "Crypto",
    "failed_attempts": "Yes"
}])
example_encoded = encoder.transform(example)
print("Example prediction:", model.predict(example_encoded)[0])
print("Class probabilities:", dict(zip(model.classes_, model.predict_proba(example_encoded)[0])))
