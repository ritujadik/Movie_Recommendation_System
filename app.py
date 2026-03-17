import streamlit as st
from FAISS import recommend_faiss  # your FAISS backend

st.set_page_config(page_title="Movie Recommendation System", layout="centered")
st.title("🎬 Movie Recommendation System")

# Input
movie_name = st.text_input("Enter Movie Name:", "")
top_n = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

# Button
if st.button("Recommend Movies"):
    if not movie_name.strip():
        st.warning("Please enter a movie title")
    else:
        recommendations = recommend_faiss(movie_name, top_n=top_n)

        # Check for error
        if recommendations and "error" in recommendations[0]:
            st.error(recommendations[0]["error"])
        else:
            st.success(f"Top {top_n} recommended movies for '{movie_name}':")

            # Display nicely in table format
            for i, movie in enumerate(recommendations, start=1):
                st.subheader(f"{i}. {movie['title']}")
                st.write(f"**Genre:** {movie['genre']}")
                st.write(f"**Director:** {movie['director']}")
                st.write(f"**Release Year:** {int(movie['release_year'])}")
                st.write("---")