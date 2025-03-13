import numpy as np
import streamlit as st
from scipy import ndimage
from PIL import Image
import io
import matplotlib.colors as mcolors

# Configuration de la page
st.set_page_config(
    page_title="VMC Visual Synth",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown(f"""
    <style>
    .main {{ background-color: #0E1117; color: #FAFAFA; }}
    .st-emotion-cache-6qob1r {{ background-color: #1A1D24 !important; }}
    h1 {{ color: #FF4B4B !important; font-family: 'Helvetica Neue', sans-serif; }}
    </style>
    """, unsafe_allow_html=True)

# Header VMC
st.image("https://i.ibb.co/0jq6Y3N/vmc-logo.png", use_container_width=300)
st.title("VMC RGB FX Processor")
st.markdown("**Synthétiseur visuel multi-effets** 🎛️🌈")

def image_to_bytes(img_array, format='JPEG'):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

# Fonctions des effets
def sobel_filter(img, mode='magnitude'):
    """Filtres Sobel avec options de direction"""
    kernels = {
        'horizontal': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'vertical': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    }
    
    if mode == 'magnitude':
        h = ndimage.convolve(img, kernels['horizontal'])
        v = ndimage.convolve(img, kernels['vertical'])
        return np.sqrt(h**2 + v**2)
    else:
        return ndimage.convolve(img, kernels[mode])

def apply_rgb_effect(rgb_img, effect_func, **kwargs):
    """Application d'effet sur chaque canal RGB"""
    channels = []
    for i in range(3):
        channel = effect_func(rgb_img[..., i], **kwargs)
        channels.append(channel)
    return np.stack(channels, axis=-1)

def grunge_effect(img, intensity=0.5):
    """Texture analogique préservant les couleurs"""
    noise = np.random.normal(loc=0, scale=intensity*0.3, size=img.shape)
    return np.clip(img + noise, 0, 1)

# Contrôles latéraux
with st.sidebar:
    st.header("Contrôles FX")
    uploaded_file = st.file_uploader("Charger une image", type=["jpg", "png", "jpeg"])
    
    # Sélection des effets
    effects = st.multiselect(
        "Effets à appliquer",
        ['Sobel Magnitude', 'Sobel Horizontal', 'Sobel Vertical', 'Texture Analog', 'Décalage Chromatique'],
        default=['Sobel Magnitude']
    )
    
    # Paramètres des effets
    params = {}
    if 'Sobel Magnitude' in effects or 'Sobel Horizontal' in effects or 'Sobel Vertical' in effects:
        params['sobel_boost'] = st.slider("Intensité Sobel", 0.1, 5.0, 1.0, 0.1)
    
    if 'Texture Analog' in effects:
        params['grunge_intensity'] = st.slider("Intensité Texture", 0.0, 1.0, 0.3)
    
    if 'Décalage Chromatique' in effects:
        params['hue_shift'] = st.slider("Décalage Hue", 0.0, 1.0, 0.0)
    
    params['output_mix'] = st.slider("Mixage Final", 0.0, 1.0, 1.0, 0.1)

# Traitement principal
if uploaded_file and effects:
    with st.spinner("Traitement en cours..."):
        # Chargement de l'image
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img).astype(float)/255.0
        result = np.copy(img_array)
        
        # Application des effets sélectionnés
        for effect in effects:
            if 'Sobel' in effect:
                mode = effect.split()[-1].lower()
                sobel_result = apply_rgb_effect(
                    img_array,
                    lambda x, m=mode: sobel_filter(x, m) * params['sobel_boost']
                )
                result = np.clip(result + sobel_result, 0, 1)
            
            if effect == 'Texture Analog':
                result = grunge_effect(result, params['grunge_intensity'])
            
            if effect == 'Décalage Chromatique':
                hsv = mcolors.rgb_to_hsv(result)
                hsv[..., 0] = (hsv[..., 0] + params['hue_shift']) % 1.0
                result = mcolors.hsv_to_rgb(hsv)
        
        # Mixage final avec l'original
        final_output = img_array * (1 - params['output_mix']) + result * params['output_mix']

    # Affichage
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.image(final_output, use_container_width=True, caption="Sortie RGB")
        
    with col2:
        st.download_button(
            "📥 Exporter",
            image_to_bytes(final_output, 'PNG'),
            file_name=f"vmc_rgb_{np.random.randint(1000,9999)}.png",
            mime="image/png"
        )
        
        # Statistiques
        st.markdown("**Canaux RGB Moyens:**")
        st.write(f"- Rouge: {final_output[..., 0].mean():.2f}")
        st.write(f"- Vert: {final_output[..., 1].mean():.2f}")
        st.write(f"- Bleu: {final_output[..., 2].mean():.2f}")

else:
    st.info("⬅️ Chargez une image et sélectionnez des effets")

# Footer
st.markdown("---")
st.markdown("**VMC Collective** - Synthèse visuelle RGB v5.0")
