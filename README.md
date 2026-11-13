# Bank Fraud Detection Using Machine Learning

An end-to-end machine learning project that detects potentially fraudulent credit card transactions using Python, Scikit-learn, Random Forest, Flask, Docker and Render.

## Project Overview

Credit card fraud detection is an important problem in financial services. Fraudulent transactions are rare compared with legitimate transactions, making this a highly imbalanced classification problem.

This project builds a machine learning model to classify transactions as:

- **0 — Legitimate**
- **1 — Fraud**

The trained model is exposed through a Flask API and a simple web interface. The application is containerized using Docker and deployed on Render.

## Live Application

[Open Bank Fraud Detection Application](https://bank-fraud-detection-tjie.onrender.com)

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib and Seaborn
- Flask
- Joblib
- Docker
- Docker Hub
- Render

## Project Workflow

1. Dataset loading and exploration
2. Data cleaning and duplicate removal
3. Exploratory data analysis
4. Train, validation and test splitting
5. Handling class imbalance
6. Training Logistic Regression, Random Forest and XGBoost
7. Model evaluation and comparison
8. Threshold selection using validation data
9. Saving the trained model using Joblib
10. Building a Flask prediction API
11. Creating a web interface
12. Containerizing the application with Docker
13. Publishing the Docker image
14. Deploying the application on Render
15. Evaluating the model on an additional external dataset

## Dataset

The original dataset contains anonymized credit card transaction features.

- 284,807 original records
- 30 input features
- 28 anonymized PCA-transformed features (`V1`–`V28`)
- Transaction time (`Time`)
- Transaction amount (`Amount`)
- Target column (`Class`)

After removing duplicate rows:

- 283,726 records remained
- 283,253 legitimate transactions
- 473 fraud transactions

The datasets are not included in this repository. Download the original dataset separately and place it inside the `data/` folder as `creditcard.csv`.

## Models Trained

Three classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Random Forest was selected for the deployed application based on its validation performance and fraud precision/recall balance.

## Model Evaluation

### Original test dataset

The selected Random Forest model was evaluated using a 40% fraud probability threshold.

| Metric | Result |
|---|---:|
| Fraud Precision | 92.21% |
| Fraud Recall | 74.74% |
| Fraud F1-score | 82.56% |
| Accuracy | 99.95% |

Confusion matrix:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,645 | 6 |
| Actual Fraud | 24 | 71 |

**Evaluation note:** The original test set was explored during earlier model and threshold experiments. Therefore, these results should be treated as exploratory rather than as a completely untouched final holdout evaluation.

### External dataset evaluation

An additional dataset containing 56,962 records was used to evaluate the saved model.

| Metric | Result |
|---|---:|
| Fraud Precision | 100.00% |
| Fraud Recall | 96.94% |
| Fraud F1-score | 98.45% |
| ROC-AUC | 0.99999982 |

Confusion matrix:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,864 | 0 |
| Actual Fraud | 3 | 95 |

No exact matches were found between the external dataset's 30 input feature records and the original dataset's feature records.

**Important external evaluation limitation:** The external dataset's README states that only transactions flagged by its original model were manually reviewed. Other transactions were automatically labelled legitimate without individual verification. This creates verification bias, meaning some actual fraud may be hidden among the legitimate labels.

Consequently, these external scores describe agreement with the dataset's recorded labels. They do not establish proven real-world banking performance.

External dataset: [Zenodo DOI 10.5281/zenodo.20030064](https://doi.org/10.5281/zenodo.20030064)

## Application Features

- Enter transaction features through a web interface.
- Submit transaction data to a Flask API.
- Receive a fraud probability.
- Classify transactions using a 40% decision threshold.
- Display the result as Fraud or Legitimate.

## Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd bank-fraud-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the trained model is available at:

```text
models/fraud_detection_rf.pkl
```

Start the application:

```bash
python src/app.py
```

Open:

```text
http://localhost:5000
```

## Docker

Build the Docker image:

```bash
docker build -t bank-fraud-detection .
```

Run the container:

```bash
docker run -p 5000:5000 bank-fraud-detection
```

Open:

```text
http://localhost:5000
```

## API Endpoint

**Endpoint:** `POST /predict`

The API accepts a JSON object containing all 30 model input features:

- `Time`
- `V1` through `V28`
- `Amount`

It returns:

- Prediction (0 or 1)
- Result (Legitimate or Fraud)
- Fraud probability

## Repository Structure

```text
bank-fraud-detection/
│
├── data/
│   └── (datasets are not included)
│
├── models/
│   └── fraud_detection_rf.pkl
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── src/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Limitations

- The original dataset uses anonymized PCA features rather than actual customer, merchant or banking fields.
- The application is an educational prototype, not a production banking fraud prevention system.
- The model requires compatible feature definitions and preprocessing.
- External evaluation labels have verification bias.
- Further testing on independently verified, representative banking data is required before real-world use.

## Future Improvements

- Add automated unit and API tests.
- Use Gunicorn as the production WSGI server.
- Add experiment tracking and model versioning.
- Improve monitoring and logging.
- Evaluate on independently verified and representative datasets.
- Explore AWS deployment.

## Author

Developed as an end-to-end machine learning and deployment portfolio project.

**Tools:** Python | Machine Learning | Flask | Docker | Render