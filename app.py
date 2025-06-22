import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Rekomendasi Drakor", page_icon="🎬", layout="centered")

# --- Load Data ---
df = pd.read_csv("kdrama_DATASET.csv")

# --- Bersihkan Nama Kolom ---
df.columns = df.columns.str.lower().str.strip()

# --- Pastikan Semua Kolom Ada ---
for col in ['description', 'genre', 'number of episodes', 'rank', 'year of release', 'url', 'poster', 'rating']:
    if col not in df.columns:
        df[col] = ''
    else:
        df[col] = df[col].fillna('')

# --- Gabungan Konten untuk TF-IDF ---
df['content'] = df['genre'] + " " + df['description']
df['title_lower'] = df['title'].str.lower().str.strip()

# --- TF-IDF & Cosine Similarity ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- Mapping Judul ke Index ---
indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()

# --- Fungsi Rekomendasi ---
def recommend(title_input, selected_genre=None, cosine_sim=cosine_sim):
    title_input = title_input.lower().strip()

    if title_input not in indices:
        return pd.DataFrame()

    idx = indices[title_input]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_k = sorted_scores[1:]  # tidak termasuk dirinya sendiri

    drama_indices = [i[0] for i in top_k]
    rec_df = df.iloc[drama_indices]

    # Filter berdasarkan genre
    if selected_genre and selected_genre != "Semua":
        rec_df = rec_df[rec_df['genre'].str.contains(selected_genre, case=False, na=False)]

    return rec_df.head(5)

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

# --- Input Judul ---
judul_list = df['title'].sort_values().tolist()
user_input = st.selectbox("Pilih judul drama Korea yang kamu suka:", options=judul_list)

# --- Filter Genre ---
genre_list = ['Semua'] + sorted(set(g.strip() for sub in df['genre'] for g in sub.split(',') if g.strip()))
selected_genre = st.selectbox("Filter berdasarkan genre (opsional):", options=genre_list)

# --- Tombol Rekomendasi ---
if st.button("Rekomendasikan 🎉"):
    with st.spinner("Mencari drama mirip..."):
        result_df = recommend(user_input, selected_genre)

    if result_df.empty:
        st.warning("❌ Tidak ditemukan rekomendasi dengan filter yang dipilih.")
    else:
        st.markdown("### Rekomendasi untukmu:")
        for _, row in result_df.iterrows():
            st.markdown(f"**🎞️ {row['title']}**")

            # Poster
            poster_url = row.get('poster', '')
            if poster_url and isinstance(poster_url, str) and poster_url.startswith("http"):
                st.image(poster_url, width=200)

            # Informasi detail
            st.markdown(f"📌 *Genre:* {row['genre']}")
            if row.get('rating', ''):
                st.markdown(f"⭐ *Rating:* {row['rating']}")
            if row.get('year of release', ''):
                st.markdown(f"🗓️ *Tahun:* {row['year of release']}")
            st.markdown(f"📝 {row['description'][:400]}...")
            if row.get('url', ''):
                st.markdown(f"[🌐 Lihat di MyDramaList]({row['url']})")

            st.markdown("---")
