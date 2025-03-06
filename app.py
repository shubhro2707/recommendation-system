from fastapi import FastAPI
import pandas as pd
import joblib
from surprise import Dataset, Reader

# Initialize FastAPI app
app = FastAPI()

# Load dataset and trained model
df_cleaned = pd.read_csv("cleaned_reviews.csv")
model = joblib.load("recommendation_model.pkl")

# Define recommendation function
def get_recommendations(user_id, n=5):
    all_products = df_cleaned['ProductId'].unique()
    predictions = [(prod, model.predict(user_id, prod).est) for prod in all_products]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [{"ProductId": prod, "EstimatedRating": rating} for prod, rating in predictions[:n]]

# Define API endpoint
@app.get("/recommend/{user_id}")
def recommend(user_id: str, n: int = 5):
    recommendations = get_recommendations(user_id, n)
    return {"user_id": user_id, "recommendations": recommendations}
