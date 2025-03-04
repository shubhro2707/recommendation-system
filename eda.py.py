import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the cleaned dataset
file_path = "C:\\Users\\mlray\\Downloads\\amazon reviews\\cleaned_reviews.csv"  # Change to your actual file path
df = pd.read_csv(file_path)

# Step 1: Check Rating Distribution
plt.figure(figsize=(8,5))
sns.countplot(x=df['Score'], palette='viridis')
plt.title("Distribution of Product Ratings", fontsize=14)
plt.xlabel("Ratings (Score)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# Step 2: Find Most Reviewed Products
top_products = df['ProductId'].value_counts().head(10)
print("\nTop 10 Most Reviewed Products:\n", top_products)

# Step 3: Find Most Active Users
top_users = df['UserId'].value_counts().head(10)
print("\nTop 10 Most Active Users:\n", top_users)

# Step 4: Analyze Review Text (WordCloud)
text_data = " ".join(df['Text'].dropna())  # Combine all reviews
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Reviews", fontsize=14)
plt.show()
