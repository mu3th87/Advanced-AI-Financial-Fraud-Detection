from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder


DATA_PATH = Path(__file__).with_name("financial_fraud_simulated_dataset.csv")
FEATURE_LABELS = {
    "amount_level": "Amount level",
    "country_risk": "Country risk",
    "device_type": "Device type",
    "login_time": "Login time",
    "merchant_type": "Merchant type",
    "failed_attempts": "Failed attempts",
}


@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_PATH)
    features = data.drop(columns=["fraud"])
    target = data["fraud"]
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded_train = encoder.fit_transform(train_features)
    encoded_test = encoder.transform(test_features)

    model = CategoricalNB(alpha=1.0)
    model.fit(encoded_train, train_target)
    predictions = model.predict(encoded_test)

    metrics = {
        "Accuracy": accuracy_score(test_target, predictions),
        "Precision": precision_score(test_target, predictions, pos_label="Yes"),
        "Recall": recall_score(test_target, predictions, pos_label="Yes"),
        "F1-score": f1_score(test_target, predictions, pos_label="Yes"),
    }
    options = {
        column: sorted(data[column].astype(str).unique().tolist())
        for column in features.columns
    }
    return model, encoder, metrics, options


st.set_page_config(
    page_title="Financial Fraud Detection",
    layout="wide",
)

model, encoder, metrics, options = train_model()

st.title("Financial Fraud Detection")
st.caption(
    "Categorical Naive Bayes model trained on a fully simulated dataset."
)

metric_columns = st.columns(4)
for column, (name, value) in zip(metric_columns, metrics.items()):
    column.metric(name, f"{value:.3f}")

st.divider()
st.subheader("Transaction details")

input_columns = st.columns(2)
selection = {}
for index, (feature, values) in enumerate(options.items()):
    with input_columns[index % 2]:
        selection[feature] = st.selectbox(FEATURE_LABELS[feature], values)

if st.button("Analyze transaction", type="primary", use_container_width=True):
    transaction = pd.DataFrame([selection])
    encoded_transaction = encoder.transform(transaction)
    prediction = model.predict(encoded_transaction)[0]
    probabilities = dict(
        zip(model.classes_, model.predict_proba(encoded_transaction)[0])
    )

    st.divider()
    if prediction == "Yes":
        st.error("Potential fraud detected")
    else:
        st.success("No fraud detected")

    fraud_probability = float(probabilities.get("Yes", 0.0))
    safe_probability = float(probabilities.get("No", 0.0))
    probability_columns = st.columns(2)
    probability_columns[0].metric(
        "Fraud probability", f"{fraud_probability:.1%}"
    )
    probability_columns[1].metric(
        "Legitimate probability", f"{safe_probability:.1%}"
    )

    chart_data = pd.DataFrame(
        {
            "Probability": [safe_probability, fraud_probability],
        },
        index=["Legitimate", "Fraud"],
    )
    st.bar_chart(chart_data)

st.info(
    "Educational demonstration only. This model must not be used for real "
    "financial decisions."
)
