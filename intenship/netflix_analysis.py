import pandas as pd
import numpy as np

# Load the dataset
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-04-20/netflix_titles.csv"
print("Loading data...")
df = pd.read_csv(url)

print("\n--- 1. Basic Understanding ---")
print(df.head())
print(df.info())
print(df.describe(include='all'))

print("\n--- 2. Content Type Distribution ---")
type_counts = df['type'].value_counts()
print(type_counts)

print("\n--- 3. Top 10 Countries with Most Content ---")
# Handling multiple countries by splitting and exploding
countries = df['country'].str.split(', ').explode()
top_countries = countries.value_counts().head(10)
print(top_countries)

print("\n--- 4. Distribution of Release Years ---")
release_year_counts = df['release_year'].value_counts().sort_index(ascending=False).head(20)
print(release_year_counts)

print("\n--- 5. Group Analysis: Growth of Movies vs TV Shows ---")
growth = df.groupby(['release_year', 'type']).size().unstack().fillna(0)
print(growth.tail(10))

print("\n--- 6. Content Ratings Analysis ---")
ratings = df['rating'].value_counts()
print(ratings)

print("\n--- 7. Duration Analysis ---")
# For Movies (extracting minutes)
movies_df = df[df['type'] == 'Movie'].copy()
movies_df['duration_min'] = movies_df['duration'].str.replace(' min', '').astype(float)
print(f"Average Movie Duration: {movies_df['duration_min'].mean():.2f} minutes")
print(f"Median Movie Duration: {movies_df['duration_min'].median()} minutes")
print(f"Standard Deviation of Duration: {movies_df['duration_min'].std():.2f}")

print("\n--- 8. Top 10 Directors on Netflix ---")
directors = df['director'].str.split(', ').explode()
top_directors = directors.value_counts().head(11) # Excluding empty if any
print(top_directors)

print("\n--- 9. Content Added Over Time ---")
df['date_added'] = pd.to_datetime(df['date_added'].str.strip())
df['year_added'] = df['date_added'].dt.year
added_per_year = df['year_added'].value_counts().sort_index()
print(added_per_year)

print("\n--- 10. Deep Dive: Content from India ---")
india_content = df[df['country'] == 'India']
india_ratings = india_content['rating'].value_counts()
print("India Ratings Distribution:")
print(india_ratings)

print("\n--- 11. Correlation Analysis ---")
# Correlation between release_year and year_added (Numeric columns)
correlation = df[['release_year', 'year_added']].corr()
print("Correlation Matrix:")
print(correlation)

