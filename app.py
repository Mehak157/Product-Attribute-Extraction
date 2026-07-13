from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

app = FastAPI(title="Product Attribute Extraction API")

# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Attributes
attributes = [
    "Silhouette",
    "Fabric",
    "Neckline",
    "Sleeve",
    "Length",
    "Embellishment",
    "Color",
    "Category"
]

# Load all models
models = {}

for attribute in attributes:
    models[attribute] = joblib.load(f"{attribute.lower()}_model.pkl")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Request body
class Product(BaseModel):
    description: str

# API endpoint
@app.post("/extract")
def extract(product: Product):

    text = clean_text(product.description)

    X = vectorizer.transform([text])

    result = {}

    for attribute in attributes:
        result[attribute] = models[attribute].predict(X)[0]

    return result