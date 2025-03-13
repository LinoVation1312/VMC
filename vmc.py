import numpy as np
import streamlit as st
from scipy import ndimage
from PIL import Image
import io
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="VMC Visual Synth",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown(f"""
    <style>
    .main {{ background-color: #0E1117; color: #FAFAFA; }}
    .st-emotion-cache-6qob1r {{ background-color: #1A1D24 !important; }}
    h1 {{ color: #FF4B4B !important; font-family: 'Helvetica Neue', sans-serif; text-shadow: 0 0 10px #FF4B4B44; }}
    .stDownloadButton button {{ background: #FF4B4B !important; border: 1px solid #FF4B4B !important; color: black !important; }}
    </style>
    """, unsafe_allow_html=True)

# Header VMC
st.image("https://i.ibb.co/0jq6Y3N/vmc-logo.png", use_container_width=300)
st.title("VMC Visual Synth")
st.markdown("**Synthétiseur visuel pour performances techno** 🌌🔊")

def image_to_bytes(img_array, format='JPEG'):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

# Nouveaux effets
def apply_distortion(img, intensity=0.1, frequency=10):
    x = np.linspace(0, frequency * 2 * np.pi, img.shape[1])
    y = np.linspace(0, frequency * 2 * np.pi, img.shape[0])
    xx, yy = np.meshgrid(x, y)
    distortion = intensity * np.sin(xx) * np.cos(yy)
    distorted = ndimage.map_coordinates(img, [np.arange(img.shape[0]) + distortion * 10, 
                                            np.arange(img.shape[1]) + distortion * 10], order=1)
    return np.clip(distorted, 0, 1)

def blend_effects(effects, mode='additive'):
    blended = np.zeros_like(effects[0])
    for effect in effects:
        if mode == 'additive':
            blended = np.clip(blended + effect, 0, 1)
        elif mode == 'multiply':
            blended = np.clip(blended * effect, 0, 1)
        elif mode == 'max':
            blended = np.maximum(blended, effect)
    return blended

with st.sidebar:
    st.header("🎛️ Contrôles Synthé")
    uploaded_file = st.file_uploader("Charger un sample visuel", type=["jpg", "png", "jpeg"])
    
    # Sélection des effets
    effect_types = st.multiselect(
        "Effets actifs",
        ['Sobel', 'Prewitt', 'Roberts', 'Distortion', 'Grayscale Edges'],
        default=['Sobel', 'Distortion'],
        format_func=lambda x: f'🔺 {x}' if 'Edges' in x else f'🌀 {x}'
    )
    
    # Paramètres généraux
    blend_mode = st.selectbox("Mode de mélange", ['additive', 'multiply', 'max'])
    output_mode = st.radio("Mode de sortie", ['Binaire', 'Niveaux de gris'])
    
    # Paramètres spécifiques
    with st.expander("Paramètres avancés"):
        distortion_intensity = st.slider("Intensité Distortion", 0.0, 1.0, 0.2)
        edge_boost = st.slider("Boost des contours", 1.0, 3.0, 1.5)
        apply_gaussian = st.checkbox("Flou Gaussien")
        if apply_gaussian:
            gaussian_sigma = st.slider("Intensité Flou", 0.0, 3.0, 1.0)

# Traitement principal
if uploaded_file and effect_types:
    with st.spinner("Syntonisation visuelle..."):
        # Préparation de l'image
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img)
        img_gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
        
        if apply_gaussian:
            img_gray = ndimage.gaussian_filter(img_gray, sigma=gaussian_sigma)
        
        # Dictionnaire d'effets
        effects = []
        kernels = {
            'Sobel': {'h': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), 
                     'v': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])},
            'Prewitt': {'h': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                       'v': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])},
            'Roberts': {'h': np.array([[1, 0], [0, -1]]),
                       'v': np.array([[0, 1], [-1, 0]])}
        }
        
        # Application des effets
        for effect in effect_types:
            if effect in kernels:
                h = ndimage.convolve(img_gray, kernels[effect]['h'], mode='nearest')
                v = ndimage.convolve(img_gray, kernels[effect]['v'], mode='nearest')
                grad = np.sqrt(h**2 + v**2) * edge_boost
                effects.append((grad - grad.min()) / (grad.max() - grad.min() + 1e-8))
            
            elif effect == 'Distortion':
                distorted = apply_distortion(img_gray, distortion_intensity)
                effects.append(distorted)
            
            elif effect == 'Grayscale Edges':
                h = ndimage.sobel(img_gray, axis=0)
                v = ndimage.sobel(img_gray, axis=1)
                grad = np.sqrt(h**2 + v**2) * edge_boost
                effects.append((grad - grad.min()) / (grad.max() - grad.min() + 1e-8))
        
        # Mélange des effets
        final_output = blend_effects(effects, blend_mode)
        
        # Post-traitement
        if output_mode == 'Binaire':
            threshold = 0.5 if len(effect_types) == 1 else 0.25 * len(effect_types)
            final_output = (final_output > threshold).astype(float)
        else:
            final_output = (final_output - final_output.min()) / (final_output.max() - final_output.min() + 1e-8)

    # Affichage
    st.header("🎚️ Sortie Synthé Visuelle", divider="red")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.image(final_output, use_container_width=True, caption=f"Mode: {blend_mode} | Effets: {', '.join(effect_types)}")
        
    with col2:
        st.download_button(
            "📥 Exporter la composition",
            image_to_bytes(final_output, 'PNG'),
            file_name=f"vmc_synth_{np.random.randint(1000,9999)}.png",
            mime="image/png"
        )
        
        # Visualisation des canaux
        st.subheader("🔍 Analyse Harmonique")
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(final_output.mean(axis=0), color='#FF4B4B')
        ax.set_facecolor('#0E1117')
        ax.grid(color='#FF4B4B33')
        st.pyplot(fig)

else:
    st.info("🌀 Connectez un sample visuel et sélectionnez des effets pour commencer")

# Footer
st.markdown("---")
st.markdown("**VMC Collective** - Système de synthèse visuelle v2.1 | Techno • Rythme • Innovation")
