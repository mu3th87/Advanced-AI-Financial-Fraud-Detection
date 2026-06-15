# Advanced AI Financial Fraud Detection

A small machine-learning project that demonstrates financial fraud detection
with a categorical Naive Bayes classifier.

The included dataset is fully simulated and does not contain real customer,
account, or transaction information.

## Project Files

- `fraud_detection_naive_bayes.py`: trains and evaluates the model.
- `financial_fraud_simulated_dataset.csv`: simulated dataset with 1,000 rows.
- `model_results_summary.txt`: saved evaluation results.
- `requirements.txt`: Python dependencies.

## Model

The script:

1. Loads the categorical transaction features.
2. Splits the data into stratified training and test sets.
3. Encodes categories with `OrdinalEncoder`.
4. Trains a `CategoricalNB` classifier.
5. Reports accuracy, precision, recall, F1-score, and a confusion matrix.
6. Predicts whether one example transaction is fraudulent.

## Run Locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies and run the project:

```bash
python -m pip install -r requirements.txt
python fraud_detection_naive_bayes.py
```

## Saved Results

Using `random_state=42`, the included run produced:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8280 |
| Precision | 0.7143 |
| Recall | 0.2885 |
| F1-score | 0.4110 |

## License

This project is available under the [MIT License](LICENSE).
