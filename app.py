import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite movie!")

# Load movie dataset
@st.cache_data
def load_data():
    data = pd.read_csv("movies.csv")
    return data

try:
    movies = load_data()

    # Check required columns
    if "title" not in movies.columns:
        st.error("The CSV file must contain a 'title' column.")
        st.stop()

    # Create a combined feature column
    feature_columns = []

    for column in ["genres", "overview", "keywords", "cast", "director"]:
        if column in movies.columns:
            feature_columns.append(column)

    if feature_columns:
        movies["combined_features"] = (
            movies[feature_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
    else:
        st.error(
            "Your dataset needs at least one of these columns: "
            "genres, overview, keywords, cast, director."
        )
        st.stop()

    # Convert text into TF-IDF features
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies["combined_features"])

    # Calculate similarity
    similarity = cosine_similarity(tfidf_matrix)

    # Movie selection
    movie_titles = movies["title"].dropna().tolist()

    selected_movie = st.selectbox(
        "🎥 Select a movie:",
        movie_titles
    )

    number_of_recommendations = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=10,
        value=5
    )

    # Recommendation button
    if st.button("🍿 Recommend Movies"):

        movie_index = movies[
            movies["title"] == selected_movie
        ].index[0]

        similarity_scores = list(
            enumerate(similarity[movie_index])
        )

        # Sort movies by similarity
        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # Get recommendations
        recommendations = similarity_scores[1:number_of_recommendations + 1]

        st.subheader("✨ Recommended Movies")

        for index, score in recommendations:
            movie_title = movies.iloc[index]["title"]

            st.write(
                f"🎬 **{movie_title}**"
            )

            st.caption(
                f"Similarity score: {score:.2f}"
            )

except FileNotFoundError:
    st.error(
        "❌ movies.csv was not found. "
        "Please upload movies.csv to the same folder as app.py."
    )

except Exception as e:
    st.error(f"Something went wrong: {e}")
