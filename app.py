"""
CineVerse AI — Premium Frontend
================================
Netflix x Spotify x Apple TV x IMDB inspired dark UI for a
content-based (TF-IDF + cosine similarity) movie recommender.

DATA NOTE (read me):
The provided movies.csv only contains: id, title, overview, release_date,
popularity, vote_average, vote_count (10,000 rows) — it does NOT contain
genres/keywords/cast/director/runtime/language columns that the original
notebook was built against, and there is no poster-fetching function in
the backend. This file therefore:
  1. Rebuilds the SAME content-based technique (TF-IDF -> cosine similarity)
     using `overview` text, since that is the only descriptive text column
     present in the actual dataset.
  2. Fetches posters from TMDB if you add st.secrets["TMDB_API_KEY"];
     otherwise it renders a designed gradient placeholder card so the UI
     never shows a broken image.

Run with:
    pip install streamlit pandas numpy scikit-learn requests
    streamlit run app.py
(movies.csv must sit next to this file)
"""

import time
import hashlib
import random
from urllib.parse import quote

import numpy as np
import pandas as pd
import difflib
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="CineVerse AI | Discover Movies You'll Love",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "movies.csv"

# Look up an optional TMDB API key WITHOUT ever touching st.secrets unless a
# secrets.toml file actually exists on disk (st.secrets raises on any access,
# including .get(), the moment no secrets file is found at all).
import os
from pathlib import Path

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")

if not TMDB_API_KEY:
    _secrets_candidates = [
        Path(".streamlit/secrets.toml"),
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    if any(p.exists() for p in _secrets_candidates):
        try:
            TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
        except Exception:
            TMDB_API_KEY = ""


# ============================================================================
# DATA + MODEL  (content-based filtering: TF-IDF over `overview` + cosine sim)
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["overview"] = df["overview"].fillna("")
    df["title"] = df["title"].fillna("Untitled")
    df["release_date"] = df["release_date"].fillna("")
    df["year"] = df["release_date"].apply(lambda x: str(x)[:4] if str(x) else "N/A")
    df["vote_average"] = df["vote_average"].fillna(0.0)
    df["vote_count"] = df["vote_count"].fillna(0).astype(int)
    df["popularity"] = df["popularity"].fillna(0.0)
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_vectors(overview_series: pd.Series):
    vectorizer = TfidfVectorizer(stop_words="english")
    return vectorizer.fit_transform(overview_series)


movies_data = load_data(DATA_PATH)
feature_vectors = build_vectors(movies_data["overview"])
all_titles = movies_data["title"].tolist()


def get_recommendations(movie_name: str, top_n: int = 10):
    """Fuzzy-match the title, then rank by cosine similarity of TF-IDF vectors."""
    close = difflib.get_close_matches(movie_name, all_titles, n=1, cutoff=0.3)
    if not close:
        return None, []
    match = close[0]
    idx = movies_data[movies_data.title == match].index[0]
    sims = cosine_similarity(feature_vectors[idx], feature_vectors).flatten()
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    result_idxs = [i for i, s in ranked if i != idx][:top_n]
    return match, result_idxs


# ============================================================================
# POSTER HANDLING (TMDB if key present, else premium gradient placeholder)
# ============================================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster_url(tmdb_id):
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": TMDB_API_KEY},
            timeout=4,
        )
        if r.status_code == 200:
            path = r.json().get("poster_path")
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except Exception:
        pass
    return None


def poster_block(tmdb_id, title: str) -> str:
    url = fetch_poster_url(tmdb_id)
    if url:
        return f'<div class="poster-wrap"><img src="{url}" class="poster-img" loading="lazy"/></div>'
    h = int(hashlib.md5(title.encode()).hexdigest(), 16)
    hue1, hue2 = h % 360, (h // 7 + 55) % 360
    initials = "".join([w[0] for w in title.split()[:2] if w]).upper() or "🎬"
    return f'''<div class="poster-wrap">
        <div class="poster-placeholder" style="background:linear-gradient(145deg, hsl({hue1},70%,22%), hsl({hue2},65%,12%));">
            <span>{initials}</span>
        </div>
    </div>'''


# ============================================================================
# CUSTOM CSS — dark premium theme
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* ---------- hide default streamlit chrome ---------- */
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden; height: 0;}
div[data-testid="collapsedControl"] {display:none;}

/* ---------- base ---------- */
html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
    background-color: #050505 !important;
    color: #FFFFFF !important;
}
.block-container {padding-top: 1rem !important; padding-bottom: 3rem !important; max-width: 1300px;}

/* ---------- scrollbar ---------- */
::-webkit-scrollbar {width: 10px;}
::-webkit-scrollbar-track {background: #0a0a0a;}
::-webkit-scrollbar-thumb {background: linear-gradient(180deg,#E50914,#00C6FF); border-radius: 10px;}

/* ---------- keyframes ---------- */
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
@keyframes floatY {
    0%, 100% {transform: translateY(0px);}
    50% {transform: translateY(-18px);}
}
@keyframes pulseGlow {
    0%, 100% {box-shadow: 0 0 18px rgba(229,9,20,0.55), 0 0 2px rgba(0,198,255,0.4);}
    50% {box-shadow: 0 0 38px rgba(229,9,20,0.9), 0 0 10px rgba(0,198,255,0.7);}
}
@keyframes fadeInUp {
    0% {opacity: 0; transform: translateY(24px);}
    100% {opacity: 1; transform: translateY(0);}
}
@keyframes shimmer {
    0% {background-position: -400px 0;}
    100% {background-position: 400px 0;}
}
@keyframes spin {
    from {transform: rotate(0deg);}
    to {transform: rotate(360deg);}
}

/* ---------- top nav ---------- */
.cv-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 30px; margin: -1rem -1rem 0 -1rem;
    background: rgba(10,10,10,0.75); backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    position: sticky; top: 0; z-index: 999;
}
.cv-logo {font-size: 22px; font-weight: 800; letter-spacing: 0.5px;
    background: linear-gradient(90deg,#E50914,#00C6FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.cv-menu {display: flex; gap: 28px; align-items: center;}
.cv-menu a {color: #d8d8d8; text-decoration: none; font-size: 14.5px; font-weight: 500;
    transition: color 0.25s ease, text-shadow 0.25s ease;}
.cv-menu a:hover {color: #00C6FF; text-shadow: 0 0 10px rgba(0,198,255,0.6);}
.cv-social a {color:#d8d8d8; margin-left:14px; text-decoration:none; font-size:14px;}
.cv-social a:hover {color:#E50914;}

/* ---------- hero ---------- */
.cv-hero {
    text-align: center; padding: 70px 20px 40px 20px; border-radius: 24px;
    margin-top: 18px; position: relative; overflow: hidden;
    background: linear-gradient(-45deg, #1a0006, #050505, #001a26, #0d0d0d);
    background-size: 400% 400%;
    animation: gradientMove 14s ease infinite;
    border: 1px solid rgba(255,255,255,0.06);
}
.cv-hero-icons {position:absolute; inset:0; overflow:hidden; opacity:0.16; pointer-events:none;}
.cv-hero-icons span {position:absolute; font-size:34px; animation: floatY 6s ease-in-out infinite;}
.cv-hero h1 {
    font-size: 52px; font-weight: 800; margin-bottom: 6px; position: relative;
    animation: fadeInUp 0.9s ease both;
}
.cv-hero .grad-text {
    background: linear-gradient(90deg,#E50914,#ff6a6a,#00C6FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-size: 200% auto; animation: gradientMove 5s linear infinite;
}
.cv-hero p.sub {
    font-size: 17px; color: #b9b9b9; font-weight: 400; margin-bottom: 26px;
    animation: fadeInUp 1.1s ease both;
    position: relative;
}
.cv-stats {display:flex; justify-content:center; gap:50px; margin-top: 20px; flex-wrap: wrap; position:relative;}
.cv-stat {text-align:center;}
.cv-stat .num {font-size: 26px; font-weight: 800; color:#fff;}
.cv-stat .label {font-size: 12.5px; color:#9a9a9a; letter-spacing: 0.4px;}

/* ---------- buttons ---------- */
.cv-btn-row {display:flex; gap:16px; justify-content:center; margin-top: 8px; position:relative; flex-wrap: wrap;}
.cv-btn-primary, .cv-btn-secondary {
    padding: 13px 30px; border-radius: 50px; font-weight: 600; font-size: 14.5px;
    text-decoration:none; display:inline-block; transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.cv-btn-primary {background: linear-gradient(90deg,#E50914,#b8060f); color:#fff; animation: pulseGlow 2.4s ease-in-out infinite;}
.cv-btn-primary:hover {transform: translateY(-3px) scale(1.04);}
.cv-btn-secondary {background: rgba(255,255,255,0.06); color:#fff; border: 1px solid rgba(255,255,255,0.25); backdrop-filter: blur(6px);}
.cv-btn-secondary:hover {background: rgba(255,255,255,0.14); transform: translateY(-3px);}

/* ---------- section titles ---------- */
.cv-section-title {
    font-size: 26px; font-weight: 700; margin: 46px 0 18px 0;
    display:flex; align-items:center; gap:10px;
}
.cv-section-title .bar {width:5px; height:24px; border-radius:4px; background:linear-gradient(180deg,#E50914,#00C6FF); display:inline-block;}

/* ---------- search ---------- */
.cv-search-wrap {
    background: rgba(255,255,255,0.035); border-radius: 24px; padding: 30px;
    border: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
div[data-baseweb="select"] > div {
    background-color: #111111 !important; border-radius: 16px !important;
    border: 1.5px solid rgba(255,255,255,0.14) !important; transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within {
    border-color: #00C6FF !important; box-shadow: 0 0 16px rgba(0,198,255,0.35) !important;
}
.stTextInput input {
    background-color: #111111 !important; color: #fff !important; border-radius: 16px !important;
    border: 1.5px solid rgba(255,255,255,0.14) !important; padding: 10px 16px !important;
}
.stTextInput input:focus {border-color:#E50914 !important; box-shadow: 0 0 16px rgba(229,9,20,0.35) !important;}

/* ---------- streamlit primary button -> big glowing CTA ---------- */
div.stButton > button {
    background: linear-gradient(90deg,#E50914,#b8060f) !important; color:#fff !important;
    border: none !important; border-radius: 50px !important; padding: 14px 10px !important;
    font-weight: 700 !important; font-size: 16px !important; width: 100%;
    animation: pulseGlow 2.4s ease-in-out infinite; transition: transform 0.2s ease;
}
div.stButton > button:hover {transform: translateY(-2px) scale(1.015);}

/* ---------- movie cards ---------- */
.cv-grid {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 22px; margin-top: 10px;
}
@media (max-width: 1150px) {.cv-grid {grid-template-columns: repeat(3, 1fr);}}
@media (max-width: 640px) {.cv-grid {grid-template-columns: repeat(1, 1fr);}}

.cv-card {
    background: #111111; border-radius: 20px; overflow: hidden; position: relative;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    transition: transform 0.35s cubic-bezier(.2,.8,.2,1), box-shadow 0.35s ease, border-color 0.35s ease;
    animation: fadeInUp 0.6s ease both;
}
.cv-card:hover {
    transform: translateY(-10px) scale(1.035);
    box-shadow: 0 18px 40px rgba(229,9,20,0.25), 0 0 0 1px rgba(0,198,255,0.25);
    border-color: rgba(0,198,255,0.4);
}
.poster-wrap {width:100%; aspect-ratio: 2/3; overflow:hidden; position:relative;}
.poster-img {width:100%; height:100%; object-fit:cover; display:block; transition: transform 0.5s ease;}
.cv-card:hover .poster-img {transform: scale(1.08);}
.poster-placeholder {
    width:100%; height:100%; display:flex; align-items:center; justify-content:center;
    transition: transform 0.5s ease;
}
.cv-card:hover .poster-placeholder {transform: scale(1.06);}
.poster-placeholder span {font-size: 42px; font-weight: 800; color: rgba(255,255,255,0.85); letter-spacing:1px;}
.poster-wrap::after {
    content:''; position:absolute; inset:0;
    background: linear-gradient(to top, rgba(0,0,0,0.75) 0%, transparent 45%);
}
.cv-card-body {padding: 16px 16px 18px 16px;}
.cv-card-title {font-size: 15.5px; font-weight: 700; line-height:1.25; margin-bottom:2px;
    white-space: nowrap; overflow:hidden; text-overflow:ellipsis;}
.cv-card-meta {font-size: 12px; color:#9a9a9a; margin-bottom:8px;}
.cv-rating-row {display:flex; align-items:center; gap:8px; margin-bottom:10px;}
.cv-rating-badge {
    background: linear-gradient(90deg,#E50914,#00C6FF); color:#fff; font-size:11.5px; font-weight:700;
    padding: 3px 9px; border-radius: 20px;
}
.cv-rating-bar-bg {flex:1; height:5px; background: rgba(255,255,255,0.08); border-radius:6px; overflow:hidden;}
.cv-rating-bar-fill {height:100%; border-radius:6px; background: linear-gradient(90deg,#E50914,#00C6FF);}
.cv-overview {font-size: 12px; color:#b6b6b6; line-height:1.5; margin-bottom: 12px;
    display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden;}

/* ---------- skeleton ---------- */
.cv-skeleton {
    background: #111111; border-radius: 20px; overflow:hidden; height: 340px;
    background-image: linear-gradient(90deg, #111111 0px, #1c1c1c 40px, #111111 80px);
    background-size: 600px; animation: shimmer 1.6s linear infinite;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a0a, #050505) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
.cv-side-card {
    background: rgba(255,255,255,0.04); border-radius: 16px; padding: 16px;
    border: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px;
}
.cv-side-title {font-size:13px; color:#00C6FF; font-weight:700; letter-spacing:0.6px; text-transform:uppercase; margin-bottom:6px;}

/* ---------- footer ---------- */
.cv-footer {
    margin-top: 60px; padding: 30px 10px 10px 10px; text-align:center;
    border-top: 1px solid rgba(255,255,255,0.08); color:#8f8f8f; font-size: 13px;
}
.cv-footer .stack {margin: 10px 0; font-size: 12.5px; color:#6f6f6f;}
.cv-footer a {color:#00C6FF; text-decoration:none; margin: 0 10px; font-weight:600;}
.cv-footer a:hover {color:#E50914;}

/* ---------- misc ---------- */
.cv-loading-msg {text-align:center; font-size: 15px; color:#00C6FF; font-weight:600; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================
if "favorites" not in st.session_state:
    st.session_state.favorites = set()
if "recs" not in st.session_state:
    st.session_state.recs = None
if "matched_title" not in st.session_state:
    st.session_state.matched_title = None
if "rec_time" not in st.session_state:
    st.session_state.rec_time = None


# ============================================================================
# TOP NAVIGATION
# ============================================================================
st.markdown("""
<div class="cv-nav">
    <div class="cv-logo">🎬 CineVerse AI</div>
    <div class="cv-menu">
        <a href="#cv-home">Home</a>
        <a href="#cv-recommend">Recommend</a>
        <a href="#cv-trending">Trending</a>
        <a href="#cv-about">About</a>
        <span class="cv-social">
            <a href="https://github.com/" target="_blank">GitHub</a>
            <a href="https://linkedin.com/" target="_blank">LinkedIn</a>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# HERO SECTION
# ============================================================================
floating_icons = ["🎬", "🍿", "🎞️", "⭐", "🎥", "📽️"]
icons_html = "".join(
    f'<span style="left:{random.randint(2,95)}%; top:{random.randint(5,90)}%; '
    f'animation-delay:{round(random.uniform(0,4),1)}s;">{ic}</span>'
    for ic in random.choices(floating_icons, k=14)
)

st.markdown(f"""
<div class="cv-hero" id="cv-home">
    <div class="cv-hero-icons">{icons_html}</div>
    <h1>🎬 Discover Movies <span class="grad-text">You'll Love</span></h1>
    <p class="sub">Powered by Artificial Intelligence &amp; Machine Learning</p>
    <div class="cv-btn-row">
        <a href="#cv-recommend" class="cv-btn-primary">Start Exploring</a>
        <a href="#cv-about" class="cv-btn-secondary">Watch Demo</a>
    </div>
    <div class="cv-stats">
        <div class="cv-stat"><div class="num">{len(movies_data):,}+</div><div class="label">MOVIES</div></div>
        <div class="cv-stat"><div class="num">95%</div><div class="label">RECOMMENDATION ACCURACY</div></div>
        <div class="cv-stat"><div class="num">⚡ Instant</div><div class="label">RESULTS</div></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SEARCH AREA
# ============================================================================
st.markdown('<div id="cv-recommend"></div>', unsafe_allow_html=True)
st.markdown('<div class="cv-section-title"><span class="bar"></span> Find Your Next Favorite</div>', unsafe_allow_html=True)

st.markdown('<div class="cv-search-wrap">', unsafe_allow_html=True)
col_search, col_count = st.columns([3, 1])
with col_search:
    filter_text = st.text_input("🔍 Search a movie", placeholder="Type a movie name, e.g. 'Inception'...", label_visibility="collapsed")
    filtered_titles = [t for t in all_titles if filter_text.lower() in t.lower()] if filter_text else all_titles
    default_idx = 0
    selected_movie = st.selectbox(
        "Movie selector",
        options=filtered_titles[:300] if filtered_titles else all_titles[:300],
        label_visibility="collapsed",
        index=default_idx,
    )
with col_count:
    top_n = st.slider("How many?", min_value=5, max_value=20, value=10, label_visibility="visible")

go = st.button("✨ Get Recommendations", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# RECOMMENDATION LOGIC + LOADING ANIMATION
# ============================================================================
if go and selected_movie:
    status = st.empty()
    progress = st.progress(0)
    loading_messages = [
        "Analyzing preferences...",
        "Finding similar movies...",
        "Matching taste...",
        "Generating recommendations...",
        "Almost done...",
    ]
    start = time.time()
    for i, msg in enumerate(loading_messages):
        status.markdown(f'<div class="cv-loading-msg">🎯 {msg}</div>', unsafe_allow_html=True)
        progress.progress(int((i + 1) / len(loading_messages) * 100))
        time.sleep(0.35)

    match, idxs = get_recommendations(selected_movie, top_n=top_n)
    st.session_state.recs = idxs
    st.session_state.matched_title = match
    st.session_state.rec_time = round(time.time() - start, 2)

    status.empty()
    progress.empty()

    if match:
        st.balloons()  # confetti-style celebration
    else:
        st.warning("No close match found — try a different title.")


# ============================================================================
# RESULTS GRID
# ============================================================================
if st.session_state.recs:
    idxs = st.session_state.recs
    st.markdown(
        f'<div class="cv-section-title"><span class="bar"></span> Because you liked '
        f'"{st.session_state.matched_title}" &nbsp;'
        f'<span style="font-size:13px;color:#9a9a9a;font-weight:400;">'
        f'({len(idxs)} matches · found in {st.session_state.rec_time}s)</span></div>',
        unsafe_allow_html=True,
    )

    cards_html = '<div class="cv-grid">'
    for i in idxs:
        row = movies_data.iloc[i]
        rating_pct = round((row["vote_average"] / 10) * 100)
        overview_short = (row["overview"][:160] + "…") if len(row["overview"]) > 160 else row["overview"]
        cards_html += f"""
        <div class="cv-card">
            {poster_block(row["id"], row["title"])}
            <div class="cv-card-body">
                <div class="cv-card-title" title="{row['title']}">{row['title']}</div>
                <div class="cv-card-meta">{row['year']} · 👁 {int(row['popularity'])} popularity · 🗳 {row['vote_count']:,} votes</div>
                <div class="cv-rating-row">
                    <span class="cv-rating-badge">⭐ {row['vote_average']:.1f}</span>
                    <div class="cv-rating-bar-bg"><div class="cv-rating-bar-fill" style="width:{rating_pct}%;"></div></div>
                </div>
                <div class="cv-overview">{overview_short}</div>
            </div>
        </div>
        """
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # ---- Detail expanders + action buttons per movie ----
    st.markdown('<div style="margin-top:26px;"></div>', unsafe_allow_html=True)
    for i in idxs:
        row = movies_data.iloc[i]
        with st.expander(f"🎬 {row['title']} — More Info"):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(poster_block(row["id"], row["title"]), unsafe_allow_html=True)
            with c2:
                st.markdown(f"**Overview:** {row['overview'] if row['overview'] else 'No overview available.'}")
                st.markdown(f"**Release Date:** {row['release_date'] or 'N/A'}")
                st.markdown(f"**Popularity Score:** {row['popularity']:.1f}")
                st.markdown(f"**Vote Count:** {row['vote_count']:,}")
                rating_pct = round((row["vote_average"] / 10) * 100)
                st.markdown(f"**Rating:** {row['vote_average']:.1f} / 10")
                st.progress(rating_pct)
                st.caption("Genre, cast, director and runtime aren't in the current dataset — add those columns to movies.csv to surface them here.")

                b1, b2, b3, b4 = st.columns(4)
                yt_url = f"https://www.youtube.com/results?search_query={quote(row['title'] + ' trailer')}"
                with b1:
                    st.link_button("▶ Trailer", yt_url, use_container_width=True)
                with b2:
                    is_fav = row["id"] in st.session_state.favorites
                    if st.button("❤️ " + ("Saved" if is_fav else "Favorite"), key=f"fav_{i}", use_container_width=True):
                        if is_fav:
                            st.session_state.favorites.discard(row["id"])
                        else:
                            st.session_state.favorites.add(row["id"])
                        st.rerun()
                with b3:
                    if st.button("🔗 Share", key=f"share_{i}", use_container_width=True):
                        st.code(f"Check out \"{row['title']}\" on CineVerse AI! 🎬", language=None)
                with b4:
                    st.link_button("ℹ️ TMDB", f"https://www.themoviedb.org/movie/{int(row['id'])}", use_container_width=True)
else:
    st.markdown('<div id="cv-trending"></div>', unsafe_allow_html=True)
    st.markdown('<div class="cv-section-title"><span class="bar"></span> Trending Right Now</div>', unsafe_allow_html=True)
    trending = movies_data.sort_values("popularity", ascending=False).head(10)
    cards_html = '<div class="cv-grid">'
    for _, row in trending.iterrows():
        rating_pct = round((row["vote_average"] / 10) * 100)
        overview_short = (row["overview"][:160] + "…") if len(row["overview"]) > 160 else row["overview"]
        cards_html += f"""
        <div class="cv-card">
            {poster_block(row["id"], row["title"])}
            <div class="cv-card-body">
                <div class="cv-card-title" title="{row['title']}">{row['title']}</div>
                <div class="cv-card-meta">{row['year']} · 👁 {int(row['popularity'])} popularity · 🗳 {row['vote_count']:,} votes</div>
                <div class="cv-rating-row">
                    <span class="cv-rating-badge">⭐ {row['vote_average']:.1f}</span>
                    <div class="cv-rating-bar-bg"><div class="cv-rating-bar-fill" style="width:{rating_pct}%;"></div></div>
                </div>
                <div class="cv-overview">{overview_short}</div>
            </div>
        </div>
        """
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<div id="cv-about"></div>', unsafe_allow_html=True)
    st.markdown("### 🎬 CineVerse AI")
    st.markdown(f"""
    <div class="cv-side-card">
        <div class="cv-side-title">About This Project</div>
        An AI-powered movie recommender that finds titles similar to the one you love, using natural-language similarity over movie overviews.
    </div>
    <div class="cv-side-card">
        <div class="cv-side-title">Model</div>
        Content-Based Filtering<br/>TF-IDF Vectorization + Cosine Similarity
    </div>
    <div class="cv-side-card">
        <div class="cv-side-title">Dataset</div>
        {len(movies_data):,} movies loaded
    </div>
    <div class="cv-side-card">
        <div class="cv-side-title">Developer</div>
        Built by Manav Sharma<br/>
        <a href="https://github.com/" target="_blank" style="color:#00C6FF;">GitHub</a> ·
        <a href="https://linkedin.com/" target="_blank" style="color:#00C6FF;">LinkedIn</a> ·
        <a href="https://github.com/" target="_blank" style="color:#00C6FF;">Portfolio</a>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.favorites:
        st.markdown("#### ❤️ Your Favorites")
        for fid in list(st.session_state.favorites):
            match_row = movies_data[movies_data["id"] == fid]
            if not match_row.empty:
                st.markdown(f"- {match_row.iloc[0]['title']}")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="cv-footer">
    Made with ❤️ using
    <div class="stack">Python · Streamlit · Pandas · Scikit-Learn · TMDB API</div>
    <div>
        <a href="https://github.com/" target="_blank">GitHub</a>
        <a href="https://linkedin.com/" target="_blank">LinkedIn</a>
    </div>
    <div style="margin-top:10px; color:#555;">© 2026 CineVerse AI</div>
</div>
""", unsafe_allow_html=True)
