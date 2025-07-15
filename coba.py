import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ================================
# Konfigurasi Halaman
# ================================
st.set_page_config(page_title="Aplikasi Skincare", layout="wide")
st.title("🧴 Aplikasi Skincare Pintar")

# ================================
# Load Model & Dataset
# ================================
model = load_model("model50.h5", compile=False)
df = pd.read_csv("https://raw.githubusercontent.com/deayulianis/skincarepintar/refs/heads/main/Skin%20Care%20Product.csv")

# ================================
# Daftar Kelas dan Mapping
# ================================
skin_classes = [
    'jerawat',
    'kulit_kering_dehidrasi',
    'kulit_kusam_noda_hitam',
    'kulit_sensitif_iritasi',
    'pori_pori_besar',
    'produksi_minyak_berlebih',
    'tanda_tanda_penuaan'
]

problem_to_effects = {
    "jerawat": [
        "Acne-Prone", "Acne-Free", "Anti-Acne", "Anti-Inflammatory", "Soothing",
        "Exfoliating", "Oil-Control", "Sebum Control", "Non-Comedogenic", "Barrier Repair"
    ],
    "kulit_kering_dehidrasi": [
        "Moisturizing", "Hydrating", "Hydrating Support", "Deep Moisture",
        "Skin Barrier Support", "Barrier Repair"
    ],
    "kulit_kusam_noda_hitam": [
        "Brightening", "Glowing", "Whitening", "Dark Spot Fading",
        "Even Skin Tone", "Antioxidant", "UV-Protection"
    ],
    "kulit_sensitif_iritasi": [
        "Sensitive-Friendly", "Calming", "Soothing", "Anti-Redness",
        "Anti-Inflammatory", "Anti-Irritation", "Skin Barrier Support", "Gentle Formula", "Ph-Balancing"
    ],
    "pori_pori_besar": [
        "Minimizing Pores", "Pore-Care", "Skin-Smoothing",
        "Exfoliating", "Sebum Control", "Non-Comedogenic"
    ],
    "produksi_minyak_berlebih": [
        "Oil-Control", "Sebum Control", "Balancing", "Exfoliating", "Ph-Balancing"
    ],
    "tanda_tanda_penuaan": [
        "Anti-Aging", "Wrinkle Care", "Firming", "Regeneration Boost",
        "Skin Barrier Support", "Deep Moisture", "Antioxidant"
    ]
}

effect_mapping = {
    "soothing & calming": "Soothing",
    "calming / soothing": "Soothing",
    "soothingglowing": "Soothing, Glowing",
    "brighteningdark spot fading": "Brightening, Dark Spot Fading",
    "anti-aging ringan": "Anti-Aging",
    "mild oil-control": "Oil-Control",
    "moisturizing.": "Moisturizing",
    "hydrating.": "Hydrating",
    "deep moistur": "Deep Moisture",
    "deep moisturee": "Deep Moisture",
    "minimizing pore": "Minimizing Pores",
    "minimizing poress": "Minimizing Pores",
    "oil-control & sebum control": "Oil-Control",
    "oil-control (indirect)": "Sebum Control",
    "pore care": "Pore-Care",
    "dark spot fading": "Dark Spot Fading",
    "uv protection": "UV-Protection",
    "anti-pollution": "Anti-Inflammatory",
    "skin skin-smoothing": "Skin-Smoothing",
    "skin-skin-smoothing": "Skin-Smoothing",
    "soothig": "Soothing",
    "combination": "Calming",
    "black-spot": "Dark Spot Fading",
    "microbiome care": "Anti-Inflammatory",
    "refreshing": "Skin Barrier Support",
    "smoothing": "Skin-Smoothing",
    "normal": "Skin Barrier Support",
    "no-whitecast": "Non-Comedogenic",
    "skin barrier repair": "Barrier Repair",
    "skin-barrier": "Barrier Repair"
}

# ================================
# Utility Functions
# ================================
# ✅ Fungsi cleaning & mapping otomatis efek
def clean_and_map_effects(effects):
    effect_list = [e.strip().lower() for e in str(effects).split(',') if e.strip()]
    result = []
    for eff in effect_list:
        eff_clean = effect_mapping.get(eff, eff).title()
        result.append(eff_clean)
    return result

df['notable_effects_clean'] = df['notable_effects'].apply(clean_and_map_effects)

#tambahan
def content_based_recommender(skin_problem, top_n=10, sort_by=None, search_query=None):
    target_effects = [e.lower() for e in problem_to_effects.get(skin_problem.lower(), [])]
    def similarity_score(row_effects):
        return len(set([e.lower() for e in row_effects]).intersection(set(target_effects)))
    df['match_score'] = df['notable_effects_clean'].apply(similarity_score)
    result = df[df['match_score'] > 0]

    # 🔍 Fitur pencarian produk
    if search_query:
        result = result[result['product_name'].str.contains(search_query, case=False, na=False)]

    # 🔃 Fitur sorting
    if sort_by == "Nama Produk":
        result = result.sort_values(by="product_name")
    elif sort_by == "Harga":
        # Buat kolom harga numerik dulu
        result['price_clean'] = result['price'].str.replace(r'[^0-9]', '', regex=True).astype(float)
        result = result.sort_values(by="price_clean")
    elif sort_by == "Merek":
        result = result.sort_values(by="brand")
    else:
        result = result.sort_values(by="match_score", ascending=False)

    return result[['product_name', 'brand', 'notable_effects', 'price', 'description', 'product_href', 'picture_src']].head(top_n)
#

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ================================
# Penjelasan Masalah Kulit
# ================================
skin_problem_info = {
    "jerawat": {
        "desc": "Merupakan kondisi peradangan pada kulit akibat pori-pori yang tersumbat oleh minyak (sebum), sel kulit mati, dan bakteri. Secara visual, ditandai dengan munculnya bintik merah, pustula, dan papula.",
        "img": "https://raw.githubusercontent.com/deayulianis/skincarepintar/refs/heads/main/Pelengkap%20Gambar%20Apk/jerawat.PNG"
    },
    "pori-pori besar": {
        "desc": "Merupakan kondisi terjadinya produksi minyak berlebih dan faktor genetik. Secara visual, ditandai dengan dengan tampilan pori-pori kulit wajah yang terlihat lebih besar atau terbuka dari biasanya terlihat di area hidung, pipi, atau dahi.",
        "img": "https://raw.githubusercontent.com/deayulianis/skincarepintar/refs/heads/main/Pelengkap%20Gambar%20Apk/pori%20besar.jpg"
    },
    "kulit kusam atau noda hitam": {
        "desc": "Merupakan kondisi disebabkan oleh penumpukan sel kulit mati, paparan sinar matahari, kurang hidrasi, atau polusi lingkungan. Secara visual, warna kulit yang tidak merata, tampak gelap, lelah, atau kurang bercahaya.",
        "img": "https://raw.githubusercontent.com/deayulianis/skincarepintar/refs/heads/main/Pelengkap%20Gambar%20Apk/kusam.jpg"
    },
    "tanda-tanda penuaan": {
        "desc": "Merupakan kondisi disebabkan karena seiring bertambahnya usia atau akibat paparan sinar UV dalam jangka panjang yang merusak struktur kolagen kulit. Secara visual, munculnya kerutan halus dan garis-garis wajah.",
        "img": "https://raw.githubusercontent.com/deayulianis/skincarepintar/refs/heads/main/Pelengkap%20Gambar%20Apk/penuaan.jpg"
    },
    "kulit sensitif atau Iritasi": {
        "desc": "Merupakan kondisi yang disebabkan kulit yang lebih tipis dan mudah bereaksi terhadap zat aktif atau cuaca ekstrem. Secara visual, kulit tampak kemerahan atau atau peradangan dan menimbulkan rasa gatal dan perih.",
        "img": "https://raw.githubusercontent.com/deayulianis/skincarepintar/refs/heads/main/Pelengkap%20Gambar%20Apk/Iritasi.jpg"
    },
    "kulit kering atau Dehidrasi": {
        "desc": "Merupakan kondisi yang disebabkan oleh kurangnya kadar air dalam lapisan kulit dan bisa dipicu oleh cuaca dingin, kurangnya kelembapan, atau sabun yang terlalu keras, Secara visual, meliputi tekstur kulit yang kasar, bersisik, terlihat pecah-pecah, dan mudah mengelupas. ",
        "img": "https://raw.githubusercontent.com/deayulianis/skincarepintar/refs/heads/main/Pelengkap%20Gambar%20Apk/kering.JPG"
    },
    "produksi minyak berlebih": {
        "desc": "Merupakan kondisi disebabkan oleh aktivitas kelenjar sebaceous yang memproduksi sebum dalam jumlah berlebihan. Secara visual, Ditunjukkan oleh permukaan wajah yang tampak mengilap, terutama di area T-zone (dahi, hidung, dan dagu).",
        "img": "https://raw.githubusercontent.com/deayulianis/skincarepintar/refs/heads/main/Pelengkap%20Gambar%20Apk/produksi%20minyak%20berlebih.jpeg"
    },
}

# ================================
# Sidebar Navigasi
# ================================
menu = st.sidebar.selectbox("📂 Pilih Menu", [
    "🏠 Home", 
    "📷 Deteksi Masalah Kulit", 
    "💡 Rekomendasi Manual", 
    "📦 Semua Produk"
])

# ================================
# Menu Tampilan
# ================================

# 🔧 Tambahkan di tampilan rekomendasi manual atau otomatis:

# 🏠 Home
if menu == "🏠 Home":
    st.subheader("📚 Penjelasan 7 Masalah Kulit Wajah")
    for key, val in skin_problem_info.items():
        st.markdown(f"### 🔹 {key.capitalize()}")
        st.image(val['img'], width=300)
        st.markdown(val['desc'])
        st.write("---")

# 📷 Deteksi otomatis
elif menu == "📷 Deteksi Masalah Kulit":
    st.subheader("📷 Deteksi Masalah Kulit + Rekomendasi")
    input_method = st.radio("Input gambar lewat:", ["Upload File", "Foto dari Kamera"])
    uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"]) if input_method == "Upload File" else st.camera_input("Ambil foto langsung")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Gambar Wajah", use_column_width=True)

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = skin_classes[np.argmax(prediction)]

        st.success(f"✅ Masalah kulit terdeteksi: **{predicted_class}**")
        st.markdown("### 💡 Rekomendasi Produk")

        top_n = st.slider("Jumlah Produk yang Ditampilkan", min_value=1, max_value=200, value=10)
        sort_option = st.selectbox("Urutkan Berdasarkan:", ["-- Relevansi --", "Nama Produk", "Harga", "Merek"])
        search_query = st.text_input("🔍 Cari Produk (opsional)", "")
        rekomendasi = content_based_recommender(predicted_class, top_n=top_n, sort_by=sort_option, search_query=search_query)
        for _, row in rekomendasi.iterrows():
            st.image(row['picture_src'], width=150)
            st.markdown(f"**{row['product_name']}** by *{row['brand']}*")
            st.markdown(f"💧 Efek: `{row['notable_effects']}`")
            st.markdown(f"💰 Harga: {row['price']}")
            st.markdown(f"[🔗 Lihat Produk]({row['product_href']})")
            st.write("---")

# 💡 Manual
elif menu == "💡 Rekomendasi Manual":
    st.subheader("💡 Rekomendasi Produk Manual")
    selected_problem = st.selectbox("Pilih masalah kulit:", list(problem_to_effects.keys()))
    
    # Pindahkan ke sini (di atas tombol!)
    top_n = st.slider("Jumlah Produk yang Ditampilkan", min_value=1, max_value=200, value=10)
    sort_option = st.selectbox("Urutkan Berdasarkan:", ["-- Relevansi --", "Nama Produk", "Harga", "Merek"])
    search_query = st.text_input("🔍 Cari Produk (opsional)", "")

    # Tombol tetap di bawah
    if st.button("Tampilkan Rekomendasi"):
        rekomendasi = content_based_recommender(
            selected_problem, 
            top_n=top_n, 
            sort_by=sort_option, 
            search_query=search_query
        )
        for _, row in rekomendasi.iterrows():
            st.image(row['picture_src'], width=150)
            st.markdown(f"**{row['product_name']}** by *{row['brand']}*")
            st.markdown(f"💧 Efek: `{row['notable_effects']}`")
            st.markdown(f"💰 Harga: {row['price']}")
            st.markdown(f"[🔗 Lihat Produk]({row['product_href']})")
            st.write("---")


# 📦 Semua produk
elif menu == "📦 Semua Produk":
    st.subheader("📦 Daftar Semua Produk")
    for _, row in df.iterrows():
        st.image(row['picture_src'], width=150)
        st.markdown(f"**{row['product_name']}** by *{row['brand']}*")
        st.markdown(f"💧 Efek: `{row['notable_effects']}`")
        st.markdown(f"💰 Harga: {row['price']}")
        st.markdown(f"[🔗 Lihat Produk]({row['product_href']})")
        st.write("---")
