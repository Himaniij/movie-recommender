import gradio as gr
import pickle
import json
import pandas as pd

# Load model and data
with open('model_small.pkl', 'rb') as f:
    model = pickle.load(f)

with open('recommendations.json') as f:
    recs = json.load(f)

movies = pd.read_csv('movie.csv')

def recommend(user_id):
    uid = str(int(user_id))
    if uid not in recs:
        return f"User {uid} not found. Try IDs between 1–500."
    
    top10 = recs[uid]
    result = "🎬 Top 10 Movie Recommendations:\n\n"
    for i, (mid, score) in enumerate(top10, 1):
        title = movies[movies['movieId'] == mid]['title'].values
        name  = title[0] if len(title) else f"Movie {mid}"
        result += f"{i}. {name}  ⭐ {score}\n"
    return result

gr.Interface(
    fn=recommend,
    inputs=gr.Number(label="Enter User ID (try 1 to 500)"),
    outputs=gr.Textbox(label="Your Recommendations", lines=12),
    title="🎬 Movie Recommender",
    description="Collaborative filtering model trained on MovieLens 20M dataset using SVD matrix factorization."
).launch()
