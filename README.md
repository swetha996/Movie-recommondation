# Content-Based Movie Recommender (MSAI-631)

A content-based movie recommendation system with a Gradio GUI. Given a movie
you liked, it suggests similar titles using TF-IDF over each film's content
(overview, genres, keywords) and cosine similarity.

## Credit / Foundation
Adapted from: <PASTE THE EXACT URL OF THE NOTEBOOK/REPO YOU FORKED>
Dataset: TMDB 5000 Movie Dataset — https://www.kaggle.com/datasets/tmdb/tmdb-5000-movie-dataset

## My modifications
- Added a Gradio GUI with a movie dropdown and a results slider
- Display similarity score (%) alongside each recommendation

## Setup
```bash
pip install -r requirements.txt
# download tmdb_5000_movies.csv from the dataset link above into this folder
python movie_recommender.py
```

## Notes on credentials
No tokens are hard-coded. If any API key is ever needed, it is read from an
environment variable by name, e.g. `os.environ.get("HF_API_TOKEN")`.
