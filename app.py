import streamlit as st
import pandas as pd
from FAISS import recommend_faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import faiss

st.set_page_config(page_title="Movie Recommendation System",layout="centered")
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter Movie Name:","")

top_n = st.slider("Number of recommendations:",min_value=1,max_value=10,value=10)

if st.button("Recommend Movies"):
    if not movie_name.strip():
        st.warning("Please enter a movie title")
    else:
        recommendations = recommend_faiss(movie_name,top_n=top_n)

        if recommendations == ["Movie not found in database"]:
            st.error("Movie not found in database")

        else:
            st.success(f"Top {top_n} recommended_movies for '{movie_name}':")
            for i,title in enumerate(recommendations,start=1):
                st.write(f"{i}. {title}")
