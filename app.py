# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Rekomendasi Drakor", page_icon="🎬", layout="centered")

# --- Load Data ---
df = pd.read_csv("kdrama_DATASET.csv")

# --- Preprocessing ---
df['description'] = df['description'].fillna('')
df['genre'] = df['genre'].fillna('')
df['poster'] = df['poster'].fillna('')  # untuk gambar
df['content'] = df['genre'] + " " + df['description']
df['title_lower'] = df['title'].str.lower().str.strip()

# --- TF-IDF Vectorization ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# --- Cosine Similarity ---
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- Mapping judul ke index ---
indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()

# --- Fungsi Rekomendasi ---
def recommend(title_input, cosine_sim=cosine_sim):
    title_input = title_input.lower().strip()

    if title_input not in indices:
        return []

    idx = indices[title_input]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_k = sorted_scores[1:6] if len(sorted_scores) > 6 else sorted_scores[1:]
    drama_indices = [i[0] for i in top_k]
    return df.iloc[drama_indices]

# --- UI ---
st.markdown("""
    <style>
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #E50914;
        text-align: center;
    }
    .sub {
        text-align: center;
        color: #888;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎬 Sistem Rekomendasi Drakor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Temukan drama Korea yang mirip dengan favoritmu 🍿</div><br>', unsafe_allow_html=True)

judul_list = df['title'].sort_values().tolist()
user_input = st.selectbox("Pilih judul drama Korea yang kamu suka:", options=judul_list)

if st.button("Rekomendasikan 🎉"):
    result_df = recommend(user_input)
    
    if result_df.empty:
        st.warning("❌ Judul tidak ditemukan atau tidak cukup data.")
    else:
        st.markdown("### Rekomendasi untukmu:")
        for _, row in result_df.iterrows():
            st.markdown(f"**🎞️ {row['title']}**")
            if row['poster']:
                st.image(row['poster'], width=200)
            st.markdown(f"📌 *Genre:* {row['genre']}")
            st.markdown(f"📝 {row['description'][:400]}...")
            st.markdown("---")
