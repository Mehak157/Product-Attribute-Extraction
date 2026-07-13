# Product-Attribute-Extraction
An NLP-based Product Attribute Extraction system using TF-IDF and Logistic Regression to predict fashion product attributes from unstructured product descriptions. The project includes model training, evaluation, and a FastAPI REST API for real-time predictions.

## Features

- NLP text preprocessing
- TF-IDF Vectorization
- Logistic Regression Classification
- Multi-Attribute Prediction
- Model Evaluation
- FastAPI REST API
- Swagger UI Documentation
- Jupyter Notebook Implementation

---

## Attributes Predicted

- Silhouette
- Fabric
- Neckline
- Sleeve
- Length
- Embellishment
- Color
- Category

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib
- FastAPI
- Uvicorn
- Jupyter Notebook


## Machine Learning Workflow

```
Dataset
     │
     ▼
Text Cleaning
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Train Logistic Regression Models
     │
     ▼
Model Evaluation
     │
     ▼
Save Models (.pkl)
     │
     ▼
FastAPI Deployment
     │
     ▼
POST /extract
     │
     ▼
JSON Response
```

---

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

## API Endpoint

### POST /extract

### Request

```json
{
  "description": "Off shoulder satin ball gown with corset bodice and sweep train in royal navy"
}
```

### Response

```json
{
  "Silhouette": "Ball Gown",
  "Fabric": "Satin",
  "Neckline": "Off Shoulder",
  "Sleeve": "Sleeveless",
  "Length": "Floor",
  "Embellishment": "Corset",
  "Color": "Royal Navy",
  "Category": "Formal"
}
```

---
##Install dependencies

```bash
pip install -r requirements.txt
```

Run the API

```bash
uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```
