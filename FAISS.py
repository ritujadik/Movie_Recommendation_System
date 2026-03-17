import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import faiss
import re
import time

app = FastAPI()
start = time.time()

# -------------------------------
# 1. Load Data
# -------------------------------
data = pd.read_csv('imdb_raw.csv')

# -------------------------------
# 2. Data Preprocessing
# -------------------------------
# Clean numeric columns
data['release_year'] = data['release_year'].str.extract(r'(\d+)').astype(float)
data['runtime'] = data['runtime'].str.extract(r'(\d+)').astype(float)
data['gross'] = data['gross'].str.extract(r'(\d+)').astype(float)

# Fill missing text values
data['genre'] = data['genre'].fillna('')
data['director'] = data['director'].fillna('')
data['title'] = data['title'].fillna('')

# Remove commas in genre/director
data['genre'] = data['genre'].str.replace(',', ' ')
data['director'] = data['director'].str.replace(',', ' ')

# Combine features
data['combine_feature'] = data['genre'] + " " + data['director']

# -------------------------------
# 3. TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['combine_feature'])

# Convert to dense + normalize for cosine similarity
tfidf_dense = tfidf_matrix.toarray().astype('float32')
tfidf_norm = normalize(tfidf_dense)

# -------------------------------
# 4. Build FAISS Index
# -------------------------------
dimension = tfidf_norm.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(tfidf_norm)

# -------------------------------
# 5. Create title index mapping
# -------------------------------
def clean_title(title):
    return re.sub(r'[^a-z0-9 ]', '', title.lower().strip())

data['title_clean'] = data['title'].apply(clean_title)
indices = pd.Series(data.index, index=data['title_clean']).drop_duplicates()

# -------------------------------
# 6. Recommendation Function
# -------------------------------
def recommend_faiss(title, top_n=5):
    title_clean = clean_title(title)

    if title_clean not in indices:
        return [{"error": "Movie not found in dataset"}]

    idx = indices[title_clean]
    query_vector = tfidf_norm[idx].reshape(1, -1)
    scores, neighbors = index.search(query_vector, top_n + 1)
    movie_indices = neighbors[0][1:]  # skip the input movie itself

    results = data.iloc[movie_indices][
        ['title', 'genre', 'director', 'release_year']
    ]

    return results.to_dict(orient='records')

# -------------------------------
# 7. FastAPI Endpoint
# -------------------------------
@app.get("/recommend")
def recommend_movies(movie: str):
    recommended = recommend_faiss(movie, top_n=5)
    return {
        "input_movie": movie,
        "recommendations": recommended
    }

# -------------------------------
# 8. Test Example
# -------------------------------
example_title = "Harry Potter and the Prisoner of Azkaban"
print(recommend_faiss(example_title, top_n=5))
end = time.time()
print("Time taken by FAISS similarity search: {:.6f}s".format(end - start))