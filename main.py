import pickle
import bz2
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
api = st.secrets["API_KEY"]

st.title('Movie Recommendation System')
with open('movies_list.pkl','rb') as f:
    movies_list = pickle.load(f)
with open('df.pkl','rb') as f:
    df = pickle.load(f)
@st.cache_data
def get_similarity_matrix(dataframe):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectorized_data = tfidf.fit_transform(dataframe['tags']).toarray()
    return cosine_similarity(vectorized_data)

similarity = get_similarity_matrix(df)
with open('df.pkl','rb') as f:
    df = pickle.load(f)
def reccomend(movie):
    movie_id = df[df['title'] == movie]['id']
    movie_index = df[df['title'] == movie]['id'].index[0]

    dist = similarity[movie_index]

    movies_list = sorted(list(enumerate(similarity[movie_index])),reverse=True,key=lambda x:x[1])[1:6]
    top5 = []
    top5_index = []
    for i in movies_list:
        top5.append(df.iloc[i[0]].title)
        top5_index.append(df.iloc[i[0]].id)
    return top5,top5_index


def get_poster(movie_id):
    try:

        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api}"


        r = requests.get(url, timeout=1)


        if r.status_code == 200:
            data = r.json()
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
        else:
            return None

    except Exception as e:
        return None
movie = st.selectbox('Select Movie',options=movies_list)


btn = st.button('Recommend Movie')
if btn:
    top5,top5_index= reccomend(movie)
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:


        poster = get_poster(top5_index[0])
        if poster is not None:
            st.image(poster)
        else:
            st.image('poster loading fail.png')
        st.text(top5[0])

    with col2:


        poster = get_poster(top5_index[1])
        if poster is not None:
            st.image(poster)
        else:
            st.image('poster loading fail.png')
        st.write(top5[1])
    with col3:


        poster = get_poster(top5_index[2])
        if poster is not None:
            st.image(poster)
        else:
            st.image('poster loading fail.png')
        st.write(top5[2])
    with col4:


        poster = get_poster(top5_index[3])
        if poster is not None:
            st.image(poster)
        else:
            st.image('poster loading fail.png')
        st.write(top5[3])

    with col5:


        poster = get_poster(top5_index[4])
        if poster is not None:
            st.image(poster)
        else:
            st.image('poster loading fail.png')
        st.write(top5[4])
