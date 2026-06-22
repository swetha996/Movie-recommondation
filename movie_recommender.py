"""
Content-Based Movie Recommender  —  MSAI-631
-------------------------------------------------------------------
A starter SKELETON. The plumbing (data loading + Gradio UI) is mostly
written for you. The parts that prove you understand the system are
left as TODOs with hints. Fill those in yourself — they are exactly
what you will explain in your report.

Foundation credit: this is adapted from the widely used content-based
approach over the TMDB 5000 Movie Dataset (Kaggle). Put the EXACT URL
of the specific notebook/repo you forked here, and in your README and
references page:
    FOUNDATION: <paste the exact URL you used>
    DATASET:    https://www.kaggle.com/datasets/tmdb/tmdb-5000-movie-dataset

Run:
    pip install -r requirements.txt
    python movie_recommender.py
"""

import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
# Download tmdb_5000_movies.csv from the Kaggle dataset and place it next
# to this file. (Keep large data OUT of git — see the .gitignore note in
# the README. Commit a tiny sample if you want the repo to run standalone.)
def load_data(path: str = "tmdb_5000_movies.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only the columns we use, and drop rows missing a title.
    df = df[["title", "overview", "genres", "keywords"]].copy()
    df = df.dropna(subset=["title"]).reset_index(drop=True)
    df["overview"] = df["overview"].fillna("")
    return df


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING  —  *** YOUR WORK STARTS HERE ***
# ----------------------------------------------------------------------
# The 'genres' and 'keywords' columns in this dataset are JSON-like
# strings, e.g.  [{"id": 28, "name": "Action"}, ...]
# You need to turn each into a clean space-separated string of names.
#
# HINT: import ast;  ast.literal_eval(text) parses the string into a list
#       of dicts, then pull out d["name"] for each dict.
def parse_names(json_like_text: str) -> str:
    import ast
    try:
        items = ast.literal_eval(json_like_text)
        return " ".join(d["name"] for d in items)
    except (ValueError, SyntaxError, TypeError):
        return ""


# Combine the cleaned features into one text "soup" per movie. The richer
# and more relevant this soup, the better the recommendations. This is a
# great place for one of your *modifications* (e.g., weight genres more
# heavily by repeating them, or fold in cast/director from credits.csv).
def build_soup(df: pd.DataFrame) -> pd.DataFrame:
    df["genres"] = df["genres"].fillna("[]").apply(parse_names)
    df["keywords"] = df["keywords"].fillna("[]").apply(parse_names)
    df["soup"] = (df["overview"] + " " + df["genres"] + " " + df["keywords"]).str.lower()
    return df


# ----------------------------------------------------------------------
# 3. VECTORIZE + SIMILARITY  —  the core of content-based filtering
# ----------------------------------------------------------------------
def build_similarity(df: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


# ----------------------------------------------------------------------
# 4. RECOMMEND
# ----------------------------------------------------------------------
def recommend(title: str, df: pd.DataFrame, sim_matrix, top_n: int = 5):
    matches = df[df["title"] == title]
    if matches.empty:
        return f"Movie '{title}' not found in the dataset."
    idx = matches.index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top = [(i, s) for i, s in scores if i != idx][:top_n]
    return [(df.iloc[i]["title"], round(s * 100, 1)) for i, s in top]


# ----------------------------------------------------------------------
# 5. GRADIO INTERFACE  (your required GUI)
# ----------------------------------------------------------------------
# This is wired up for you. Once the functions above work, it runs as-is.
def main():
    df = load_data()
    df = build_soup(df)
    sim = build_similarity(df)
    titles = sorted(df["title"].tolist())

    def ui_recommend(title, count):
        if not title:
            return "Pick a movie first."
        recs = recommend(title, df, sim, top_n=int(count))
        if isinstance(recs, list):
            return "\n".join(f"{i+1}. {t} ({s}% match)" for i, (t, s) in enumerate(recs))
        return recs

    demo = gr.Interface(
        fn=ui_recommend,
        inputs=[
            gr.Dropdown(choices=titles, label="Pick a movie you liked"),
            gr.Slider(1, 10, value=5, step=1, label="How many recommendations?"),
        ],
        outputs=gr.Textbox(label="You might also like"),
        title="Content-Based Movie Recommender",
        description="Suggests movies similar to one you choose, based on its content.",
    )
    demo.launch()


if __name__ == "__main__":
    main()


# ======================================================================
# EXTENSION IDEAS  — pick at least TWO and describe them in your report's
# "Modifications Made" section. Each one demonstrates that you understand
# how the system works:
#   A. Replace the original console/notebook interface with this Gradio
#      GUI (already a modification if your foundation used neither).
#   B. Enrich the soup with cast + director from credits.csv.
#   C. Add a genre filter so results are restricted to a chosen genre.
#   D. Show the similarity score (%) next to each recommendation.
#   E. Add a CountVectorizer-vs-TfidfVectorizer toggle and compare results.
#   F. Cache the similarity matrix to disk so startup is faster.
# ======================================================================
