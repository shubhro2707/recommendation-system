import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import joblib

# Load cleaned dataset
df_cleaned = pd.read_csv("cleaned_reviews.csv")  # Ensure this file is in the same directory

# Define rating scale for Surprise library
reader = Reader(rating_scale=(1, 5))

# Load data into Surprise format
data = Dataset.load_from_df(df_cleaned[['UserId', 'ProductId', 'Score']], reader)

# Use Singular Value Decomposition (SVD) for recommendations
model = SVD()

# Perform cross-validation
cross_validate(model, data, cv=5, verbose=True)

# Train the model on the full dataset
trainset = data.build_full_trainset()
model.fit(trainset)

# Function to get top-N recommendations for a user
def get_recommendations(user_id, n=5):
    all_products = df_cleaned['ProductId'].unique()
    predictions = [(prod, model.predict(user_id, prod).est) for prod in all_products]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

# Test recommendations for a random user
user_sample = df_cleaned['UserId'].sample(1).values[0]
recommended_products = get_recommendations(user_sample)
print(f"Top recommended products for User {user_sample}: {recommended_products}")



# Save trained model
joblib.dump(model, "recommendation_model.pkl")
print("Model saved successfully!")

