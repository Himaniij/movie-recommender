import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Movie Recommender Dashboard", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommender — Analytics Dashboard")
st.markdown("Built on MovieLens 20M dataset · SVD Collaborative Filtering")

# Load data
genre_counts = pd.read_csv('genre_counts.csv')
rating_dist  = pd.read_csv('rating_dist.csv')
top_movies   = pd.read_csv('top_movies.csv')

# Row 1 — scorecards
col1, col2, col3 = st.columns(3)
col1.metric("Total Ratings",  "20,000,263")
col2.metric("Total Movies",   "27,278")
col3.metric("Total Users",    "138,493")

st.divider()

# Row 2 — charts
col4, col5 = st.columns(2)

with col4:
    st.subheader("📊 Ratings Distribution")
    fig1 = px.bar(rating_dist, x='rating', y='count',
                  color='count', color_continuous_scale='Blues')
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    st.subheader("🎭 Top Genres")
    fig2 = px.bar(genre_counts.head(10), x='count', y='genre',
                  orientation='h', color='count',
                  color_continuous_scale='Purples')
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# Row 3 — top movies
st.subheader("⭐ Top 10 Highest Rated Movies (min 500 ratings)")
fig3 = px.bar(top_movies, x='avg_rating', y='title',
              orientation='h', color='avg_rating',
              color_continuous_scale='Greens')
st.plotly_chart(fig3, use_container_width=True)

st.caption("Project by Himani Joshi · MovieLens 20M · SVD Matrix Factorization · Deployed on Streamlit Cloud")
