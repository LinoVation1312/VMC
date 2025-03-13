import numpy as np
import streamlit as st
from scipy import ndimage
from PIL import Image
import io
import matplotlib.pyplot as plt

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

# Fonctions des nouveaux effets
def apply_edge_control(img, boost=1.0, thickness=1.0):
    """Contrôle des contours avec gestion séparée de l'intensité et de l'épaisseur"""
    # Détection des contours
    h = ndimage.sobel(img, axis=0)
    v = ndimage.sobel(img, axis=1)
    edges = np.hypot(h, v)
    
    # Application du boost avant normalisation
    edges = ndimage.gaussian_filter(edges * boost, sigma=thickness)
    
    # Normalisation adaptative
    p99 = np.percentile(edges, 99)
    return np.clip(edges / (p99 + 1e-8), 0, 1)

def apply_grunge_effect(img, intensity=0.5):
    """Effet de texture analogique"""
    noise = np.random.normal(loc=0, scale=intensity, size=img.shape)
    return np.clip(img + noise, 0, 1)

def apply_color_shift(img, hue=0.0):
    """Décalage chromatique pour effets stroboscopiques"""
    hsv = plt.colors.rgb_to_hsv(img)
    hsv[..., 0] = (hsv[..., 0] + hue) % 1.0
    return plt.colors.hsv_to_rgb(hsv)

# Contrôles latéraux
with st.sidebar:
    st.header("Contrôles FX")
    uploaded_file = st.file_uploader("Charger une image", type=["jpg", "png", "jpeg"])
    
    # Paramètres principaux
    edge_boost = st.slider("Intensité contours", 0.1, 5.0, 2.0, 0.1)
    edge_thickness = st.slider("Épaisseur contours", 0.5, 5.0, 1.5, 0.1)
    grunge_intensity = st.slider("Texture analogique", 0.0, 1.0, 0.2)
    color_shift = st.slider("Décalage chromatique", 0.0, 1.0, 0.0)
    output_mix = st.slider("Mixage final", 0.0, 1.0, 1.0, 0.1)

# Traitement principal
if uploaded_file:
    with st.spinner("Processing..."):
        # Chargement de l'image
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img).astype(float)/255.0
        
        # Conversion en HSV pour les effets chromatiques
        hsv = plt.colors.rgb_to_hsv(img_array[..., :3])
        
        # Application des effets
        processed = apply_edge_control(hsv[..., 2], edge_boost, edge_thickness)
        processed = apply_grunge_effect(processed, grunge_intensity)
        
        # Application du décalage chromatique
        hsv[..., 0] = (hsv[..., 0] + color_shift) % 1.0
        hsv[..., 2] = np.clip(hsv[..., 2] * (1 - output_mix) + processed * output_mix, 0, 1)
        
        # Conversion finale en RGB
        final_output = plt.colors.hsv_to_rgb(hsv)

    # Affichage des résultats
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.image(final_output, use_container_width=True, caption="Sortie FX")
        
    with col2:
        st.download_button(
            "📥 Exporter",
            image_to_bytes(final_output, 'PNG'),
            file_name=f"vmc_fx_{np.random.randint(1000,9999)}.png",
            mime="image/png"
        )
        
        # Visualisation des paramètres
        st.markdown("**Paramètres actifs:**")
        st.write(f"- Boost contours: {edge_boost}x")
        st.write(f"- Épaisseur: {edge_thickness}px")
        st.write(f"- Texture: {grunge_intensity*100}%")
        st.write(f"- Chroma: {color_shift*360:.0f}°")

else:
    st.info("⬅️ Chargez une image pour commencer")

# Footer
st.markdown("---")
st.markdown("**VMC Collective** - Interface FX v3.2 | Techno • Réactivité • Intensité")
