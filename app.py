# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(page_title="Rekomendasi Drakor", page_icon="🎬", layout="centered", initial_sidebar_state="collapsed")

# --- Load Data ---
df = pd.read_csv("kdrama_DATASET.csv")
df['description'] = df['description'].fillna('')
df['genre'] = df['genre'].fillna('')
df['content'] = df['genre'] + " " + df['description']
df['title_lower'] = df['title'].str.lower().str.strip()

# --- TF-IDF Vectorization ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# --- Cosine Similarity antar drama ---
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- Mapping judul ke index dataframe ---
indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()

# --- Fungsi Rekomendasi ---
def recommend(title_input, cosine_sim=cosine_sim):
    title_input = title_input.lower().strip()

    if title_input not in indices:
        return ["❌ Judul tidak ditemukan. Coba cek ejaannya."]
    
    idx = indices[title_input]
    sim_scores = list(enumerate(cosine_sim[idx]))

    if len(sim_scores) <= 1:
        return ["❌ Tidak cukup data untuk merekomendasikan drama lain."]

    # Cek dulu panjang data sebelum slicing
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Jika jumlah data kurang dari 6, ambil semampunya
    top_k = sorted_scores[1:6] if len(sorted_scores) > 6 else sorted_scores[1:]
    
    drama_indices = [i[0] for i in top_k]
    return df['title'].iloc[drama_indices].tolist()

# --- UI Streamlit ---
st.markdown(
    """
    <style>
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #E50914;
        text-align: center;
    }
    .sub {
        text-align: center;
        color: #ccc;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🎬 Sistem Rekomendasi Drakor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Temukan drama Korea yang mirip dengan favoritmu 🍿</div><br>', unsafe_allow_html=True)

# Dropdown biar tidak typo
judul_list = df['title'].sort_values().tolist()
user_input = st.selectbox("Pilih judul drama Korea yang kamu suka:", options=judul_list)

if st.button("Rekomendasikan 🎉"):
    recommendations = recommend(user_input)
    if recommendations and "❌" in recommendations[0]:
        st.warning(recommendations[0])
    else:
        st.markdown("### Rekomendasi untukmu:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. **{rec}**")
