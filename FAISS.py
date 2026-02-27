import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import time
import faiss
import numpy as np

from main import tfidf_vectorizer, tfidf_matrix

app = FastAPI()

start = time.time()
#1. data loading
data = pd.read_csv('imdb_raw.csv')
print(data.columns)
print(data.head(10))

#2.Check the missing value
print(data.isnull().sum())

#3.Data Preprocessing
print(data.info())
"""As per the data information we have few columns which are not in correct format like release year should be int instead of str,runtime should be numeric instead of str,gross should be numeric instead of str,"""
# convert the release date into int
data['release_year'] = data['release_year'].str.extract('(\d+)')
data['release_year'] = data['release_year'].astype(int)
data['release_year'].head()
print(data.dtypes)
# convert the runtime
data['runtime'] = data['runtime'].str.extract('(\d+)')
data['runtime'] = data['runtime'].astype(int)
data['runtime'].head()
print(data.dtypes)

# convert the gross
data['gross'] = data['gross'].str.extract('(\d+)')
data['gross'] = data['gross'].astype(int)
data['gross'].head()
print(data.dtypes)

# Feature Engineering
# step-1: Select important features
"""make the strongest feature by adding  genre + director name"""
# step-2 : Create a new column for the  same
data['genre'] = data['genre'].str.replace(',','')
data['director'] = data['director'].str.replace(',','')
data['combine_feature'] = data['genre'] + " " + data['director']
print(data['combine_feature'].head(5))

"""data vectorization and similarity computation"""
# Convert sparse TF-IDF to dense float32 and normal size
tfidf_dense = np.asarray(tfidf_matrix.todense())
tfidf_norm = normalize(tfidf_dense)
# build the FAISS Index
d = tfidf_norm.shape[1]  # dimension
index = faiss.IndexFlatIP(d) # inner product-cosine similarity
index.add(np.array(tfidf_norm))

""" 
above command will create the each movie in a vector
each word in combine_feature gets a weight based on TF-IDF
The result tfidf_matrix is numeric and ready for similarity calculation
"""
# check the shape
print(tfidf_matrix.shape)

# Step-2 Compute Similarity
cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
print(cosine_sim)

# Map movie titles to the index
indices = pd.Series(data.index, index=data['title']).drop_duplicates()

# Build the FAISS index
d = tfidf_norm.shape[1]
index = faiss.IndexFlatIP(d)
index.add(np.array(tfidf_norm))

# Map titles to indices
indices = pd.Series(data.index, index=data['title']).drop_duplicates()

"""Build the recommendation function"""


def recommend_faiss(title, top_n=5):
    title = title.strip()
    if title not in indices:
        return ["Movie not found in the dataset"]

    idx = indices[title]
    query_vector = tfidf_norm[idx].reshape(1, -1).astype('float32')
    D, I = index.search(query_vector, top_n + 1)
    movie_indices = I.flatten()[1:]  # skip itself

    # return as a list instead of Series
    return data['title'].iloc[movie_indices].tolist()

# def recommend_movies_with_info(title,top_n=5):
#     recommended_titles = recommendation(title,top_n=top_n)
#     recommendation_movies = data[data['title'].isin(recommended_titles)]
#     return recommendation_movies[['title','genre','director','release_year','rating']]


@app.get("/recommend")
def recommend_movies(movie:str):
    recommended = recommend_faiss(movie,top_n=5)
    return recommended.tolist()

end = time.time()
print(recommend_faiss(" Harry Potter and the Prisoner of Azkaban",top_n=20))
print("Time taken by Sklearn Cosine Similarity:{:.6f}s".format(end-start))