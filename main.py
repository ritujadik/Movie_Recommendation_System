import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time



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
tfidf_vectorizer = TfidfVectorizer()
# create the TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english') # ignore common words
tfidf_matrix = tfidf.fit_transform(data['combine_feature'])
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

"""Build the recommendation function"""
def recommendation(title,cosine_sim=cosine_sim,data=data,indices=indices,top_n=5):
    # get the index of the movie which match the title
    idx = indices[title]
    # get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # sort the movie based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # get the scores of the top n most similar movies (skip the first one as it is the same movie)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    # return the top_n most similar movie titles
    return data['title'].iloc[movie_indices]

def recommend_movies_with_info(title,top_n=5):
    recommended_titles = recommendation(title,top_n=top_n)
    recommendation_movies = data[data['title'].isin(recommended_titles)]
    return recommendation_movies[['title','genre','director','release_year','rating']]

end = time.time()
print(recommend_movies_with_info("Harry Potter and the Prisoner of Azkaban",top_n=5))
print("Time taken by Sklearn Cosine Similarity:{:.6f}s".format(end-start))