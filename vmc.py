import numpy as np
import streamlit as st
from scipy import ndimage
from PIL import Image
import io
import matplotlib.colors as mcolors

# Configuration de la page
st.set_page_config(
    page_title="VMC Visual Processor",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS minimaliste
st.markdown(f"""
    <style>
    .main {{ background-color: #0E1117; color: #FAFAFA; }}
    .st-emotion-cache-6qob1r {{ background-color: #1A1D24 !important; }}
    h1 {{ color: #FF4B4B !important; font-family: 'Helvetica Neue', sans-serif; }}
    .stSlider {{ max-width: 300px; }}
    </style>
    """, unsafe_allow_html=True)

# Header VMC
st.image("https://i.ibb.co/0jq6Y3N/vmc-logo.png", use_container_width=300)
st.title("VMC FX Processor")
st.markdown("**Interface de traitement visuel instantané** 🎛️⚡")

def image_to_bytes(img_array, format='JPEG'):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

def apply_edge_control(img, boost=1.0, thickness=1.0):
    """Contrôle précis des contours"""
    # Détection des contours
    h = ndimage.sobel(img, axis=0)
    v = ndimage.sobel(img, axis=1)
    edges = np.hypot(h, v)
    
    # Application des paramètres
    edges = ndimage.gaussian_filter(edges * boost, sigma=thickness*2)
    
    # Normalisation adaptative
    p99 = np.percentile(edges, 99)
    return np.clip(edges / (p99 + 1e-8), 0, 1)

def apply_grunge_effect(img, intensity=0.5):
    """Ajout de texture analogique"""
    noise = np.random.normal(loc=0, scale=intensity*0.3, size=img.shape)
    return np.clip(img + noise, 0, 1)

def apply_color_shift(rgb_img, hue_shift=0.0):
    """Décalage chromatique HSV"""
    hsv = mcolors.rgb_to_hsv(rgb_img)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 1.0
    return mcolors.hsv_to_rgb(hsv)

# Contrôles latéraux
with st.sidebar:
    st.header("Contrôles FX")
    uploaded_file = st.file_uploader("Charger une image", type=["jpg", "png", "jpeg"])
    
    # Paramètres principaux
    edge_boost = st.slider("INTENSITÉ CONTOURS", 0.1, 5.0, 2.0, 0.1)
    edge_thickness = st.slider("ÉPAISSEUR CONTOURS", 0.0, 5.0, 1.5, 0.1)
    grunge_intensity = st.slider("TEXTURE ANALOG", 0.0, 1.0, 0.3)
    hue_shift = st.slider("DÉCALAGE CHROMATIQUE", 0.0, 1.0, 0.0)
    output_mix = st.slider("MIXAGE FINAL", 0.0, 1.0, 1.0, 0.1)

# Traitement principal
if uploaded_file:
    with st.spinner("PROCESSING..."):
        # Chargement et conversion
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img).astype(float)/255.0
        
        # Application des effets
        edges = apply_edge_control(img_array.mean(axis=2), edge_boost, edge_thickness)
        textured = apply_grunge_effect(edges, grunge_intensity)
        colored = apply_color_shift(img_array, hue_shift)
        
        # Mixage final
        final = img_array * (1 - output_mix) + textured[..., None] * output_mix
        final = np.clip(final, 0, 1)

    # Affichage
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.image(final, use_container_width=True, caption="SORTIE VMC")
        
    with col2:
        st.download_button(
            "📥 EXPORTER",
            image_to_bytes(final, 'PNG'),
            file_name=f"vmc_export_{np.random.randint(1000,9999)}.png",
            mime="image/png"
        )
        
        # Visualisation des paramètres
        st.markdown("**PARAMÈTRES ACTIFS:**")
        st.write(f"- Boost: x{edge_boost:.1f}")
        st.write(f"- Épaisseur: {edge_thickness:.1f}px")
        st.write(f"- Texture: {grunge_intensity*100:.0f}%")
        st.write(f"- Chroma: {hue_shift*360:.0f}°")

else:
    st.info("⬅️ CHARGEZ UNE IMAGE POUR COMMENCER")

# Footer
st.markdown("---")
st.markdown("**VMC COLLECTIVE** - SYSTÈME DE SYNTHÈSE VISUELLE v4.0")
