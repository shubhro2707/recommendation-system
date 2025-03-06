import pandas as pd

# Load the dataset
file_path = "Reviews.csv"  # Change to your file path
df = pd.read_csv(file_path)

# Step 1: Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Step 2: Drop unnecessary columns
df_cleaned = df.drop(columns=['Id', 'ProfileName', 'Summary'])

# Step 3: Convert 'Time' column to datetime
df_cleaned['Time'] = pd.to_datetime(df_cleaned['Time'], unit='s')

# Step 4: Remove users with fewer than 5 reviews (cold-start problem)
user_review_counts = df_cleaned['UserId'].value_counts()
active_users = user_review_counts[user_review_counts >= 5].index
df_cleaned = df_cleaned[df_cleaned['UserId'].isin(active_users)]

# Step 5: Remove products with fewer than 5 reviews
product_review_counts = df_cleaned['ProductId'].value_counts()
popular_products = product_review_counts[product_review_counts >= 5].index
df_cleaned = df_cleaned[df_cleaned['ProductId'].isin(popular_products)]

# Step 6: Normalize ratings (convert Score to float)
df_cleaned['Score'] = df_cleaned['Score'].astype(float)

# Step 7: Save the cleaned dataset
df_cleaned.to_csv("cleaned_reviews.csv", index=False)

# Display final dataset info
print("Final Dataset Info:")
print(df_cleaned.info())

# Display first few rows
print(df_cleaned.head())
