import pickle

import pandas as pd
import streamlit as st


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        # fetch the movie poster
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names


st.header('Movie Recommender System')
movies = pickle.load(open('movies.pkl','rb'))
movies = pd.DataFrame(movies)
similarity = pickle.load(open('similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    for i in recommended_movie_names:
        st.write(i)