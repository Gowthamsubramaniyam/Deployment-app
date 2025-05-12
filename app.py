import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = {
    "title": [
        "Inception", "Interstellar", "The Dark Knight", "The Prestige", "Memento",
        "The Matrix", "Shutter Island", "Fight Club", "The Truman Show", "Donnie Darko"
    ],
    "description": [
        "A thief who steals corporate secrets through the use of dream-sharing technology.",
        "A team of explorers travel through a wormhole in space to ensure humanity's survival.",
        "Batman sets out to dismantle the remaining criminal organizations that plague Gotham City.",
        "Two rival stage magicians engage in a battle to create the ultimate illusion while sacrificing everything.",
        "A man with short-term memory loss uses tattoos and notes to hunt for his wife’s killer.",
        "A hacker discovers that the world he lives in is a simulation and joins a rebellion.",
        "A U.S. Marshal investigates the disappearance of a prisoner from a hospital for the criminally insane.",
        "An insomniac office worker and a soapmaker form an underground fight club that evolves into something much more.",
        "An insurance salesman discovers his whole life is a TV show and struggles with reality.",
        "A troubled teenager is plagued by visions of a man in a rabbit suit who manipulates him to commit crimes."
    ]
}

df = pd.DataFrame(data)
df['description'] = df['description'].str.lower()

# TF-IDF and similarity
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(title, top_n=5):
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [df['title'][i[0]] for i in sim_scores[1:top_n+1]]
    return top_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Select a movie:", df['title'].values)

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)
    st.subheader("You may also like:")
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")
