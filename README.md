🎬 **Movie Recommendation System**
**A content-based movie recommendation system built using FAISS, TF-IDF, FastAPI, and Streamlit. Users can input a movie and get top similar movies based on genre and director, with a clean UI.

**Deployed link-"https://movierecommendationsystem-g96sssvkgkkpgujzpeux8h.streamlit.app/"**
**Features**
Recommends movies based on content similarity.
Uses TF-IDF to vectorize movie features (genre + director).
FAISS for fast similarity search.
Clean Streamlit UI for interactive experience.
Optional: fuzzy matching for approximate movie title searches.**
**Screenshot**
**<img width="903" height="888" alt="Movie_Recommendation_System" src="https://github.com/user-attachments/assets/79452e68-9d76-4b5e-a34b-0e44613d69ac" />
**Installation**
clone the repo-git clone https://github.com/ritujadik/Movie_Recommendation_System.git
                         cd Movie_Recommendation_System
create the virtual environment-python -m venv venv
                               venv\Scripts\activate 
Install dependencies-pip install -r requirements.txt
Run locally-streamlit run app.py-Open http://localhost:8501 in browser.
uvicorn FAISS:app --reload-for Backend 

**Project Structure**
Movie_Recommendation_System/
├─ FAISS.py                 # FastAPI backend + recommend_faiss function
├─ app.py                   # Streamlit UI
├─ imdb_raw.csv             # Dataset
├─ requirements.txt         # Dependencies
├─ README.md                # This file
**How It Works**
Data Preprocessing: Clean the dataset (genre, director, release year).
Feature Engineering: Combine genre + director as the strongest content feature.
TF-IDF Vectorization: Convert movie features into numeric vectors.
FAISS Indexing: Build FAISS index for fast similarity search.
**Recommendation Function:**
Takes a movie title as input
Finds its vector in FAISS
Returns top N similar movies with details (title, genre, director, release year)
**Streamlit UI:**
enters a movie name
UI displays recommendations with clean formatting

**Technologies Used**
Python 3.10+
FAISS-fast similarity search
scikit-learn—TF-IDF, normalization
FastAPI— backend API
Streamlit — interactive UI

**Created By-Rituja Dikshit**
feel free to reach out for questions or contributions.
                         

