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
    .stSlider {{ max-width: 300px; }}
    [data-testid="stExpander"] {{ border-color: #FF4B4B55 !important; }}
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

# Nouveaux effets
def apply_emboss(img, intensity=1.0):
    kernel = np.array([[-2*intensity, -intensity, 0], 
                       [-intensity, 1, intensity], 
                       [0, intensity, 2*intensity]])
    return ndimage.convolve(img, kernel)

def edge_thickness_control(edges, thickness=1):
    return ndimage.gaussian_filter(edges.astype(float), sigma=thickness)

def color_invert(img):
    return 1.0 - img

# Contrôles latéraux
with st.sidebar:
    st.header("Contrôles FX")
    uploaded_file = st.file_uploader("Charger une image", type=["jpg", "png", "jpeg"])
    
    # Sélection des effets principaux
    fx_selection = st.multiselect(
        "Effets principaux",
        ['Contours Sobel', 'Distortion', 'Emboss', 'Inversion'],
        default=['Contours Sobel']
    )
    
    # Paramètres universels
    edge_boost = st.slider("Intensité contours", 0.5, 3.0, 1.5, 0.1)
    distortion_strength = st.slider("Force Distortion", 0.0, 1.0, 0.3) if 'Distortion' in fx_selection else 0.0
    emboss_intensity = st.slider("Intensité Emboss", 0.5, 2.0, 1.0) if 'Emboss' in fx_selection else 1.0
    edge_thickness = st.slider("Épaisseur contours", 0.5, 3.0, 1.0, 0.5)
    output_mix = st.slider("Mixage final", 0.0, 1.0, 1.0, 0.1)
    apply_gaussian = st.checkbox("Pré-flou", value=True)
    gaussian_sigma = st.slider("Intensité flou", 0.0, 2.0, 0.8)

# Traitement principal
if uploaded_file and fx_selection:
    with st.spinner("Processing..."):
        # Préparation de l'image
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img)
        img_gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
        
        # Pré-traitement
        if apply_gaussian:
            img_gray = ndimage.gaussian_filter(img_gray, sigma=gaussian_sigma)
        
        # Application des effets
        result = np.copy(img_gray)
        
        # Contours Sobel
        if 'Contours Sobel' in fx_selection:
            h = ndimage.sobel(result, axis=0)
            v = ndimage.sobel(result, axis=1)
            edges = np.sqrt(h**2 + v**2) * edge_boost
            edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
            edges = edge_thickness_control(edges, edge_thickness)
            result = edges
        
        # Distortion
        if 'Distortion' in fx_selection:
            x = np.linspace(0, 8 * np.pi, result.shape[1])
            y = np.linspace(0, 8 * np.pi, result.shape[0])
            xx, yy = np.meshgrid(x, y)
            distortion = distortion_strength * (np.sin(xx) * np.cos(yy))
            result = ndimage.map_coordinates(result, 
                [np.arange(result.shape[0]) + distortion*5, 
                 np.arange(result.shape[1]) + distortion*5], 
                order=1)
        
        # Emboss
        if 'Emboss' in fx_selection:
            embossed = apply_emboss(result, emboss_intensity)
            result = np.clip((result * 0.5 + embossed * 0.5), 0, 1)
        
        # Inversion
        if 'Inversion' in fx_selection:
            result = color_invert(result)
        
        # Mixage final
        result = np.clip(result * output_mix + img_gray * (1 - output_mix), 0, 1)

    # Affichage des résultats
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.image(result, use_container_width=True, caption="Sortie FX")
        
    with col2:
        st.download_button(
            "📥 Exporter",
            image_to_bytes(result, 'PNG'),
            file_name=f"vmc_fx_{np.random.randint(1000,9999)}.png",
            mime="image/png"
        )
        
        # Analyse en temps réel
        st.markdown("**Analyse de signal**")
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.hist(result.ravel(), bins=50, color='#FF4B4B')
        ax.set_facecolor('#0E1117')
        plt.xticks([])
        plt.yticks([])
        st.pyplot(fig)

else:
    st.info("⬅️ Chargez une image et sélectionnez des effets pour commencer")

# Footer
st.markdown("---")
st.markdown("**VMC Collective** - Interface FX v3.0 | Techno • Minimalisme • Performance")
