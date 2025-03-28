import numpy as np
import streamlit as st
from scipy import ndimage
from PIL import Image, ImageOps
import io
import matplotlib.colors as mcolors
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="VMC Ultimate Synth",
    page_icon="🎛️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Paramètres de performance
MAX_IMAGE_SIZE = 2000
MAX_FILE_SIZE_MB = 10
QUALITY = 85
PROCESSING_LIMIT = 4000

# Style CSS amélioré
st.markdown(f"""
    <style>
    .main {{ background-color: #0E1117; color: #FAFAFA; }}
    .st-emotion-cache-6qob1r {{ background-color: #1A1D24 !important; }}
    h1 {{ color: #FF4B4B !important; font-family: 'Helvetica Neue', sans-serif; }}
    .warning {{ color: #FF4B4B !important; font-weight: bold; }}
    .info {{ color: #4BFF4B !important; }}
    .stProgress > div > div > div > div {{
        background-color: #FF4B4B !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Header VMC
st.image("https://m.media-amazon.com/images/M/MV5BYmZlOTY2OGUtYWY2Yy00NGE0LTg5YmQtNmM2MmYxOWI2YmJiXkEyXkFqcGdeQXVyMTkxNjUyNQ@@._V1_.jpg", use_column_width=True)
st.title("VMC Ultimate FX Processor")
st.markdown("**Station de traitement visuel multi-effets** 🎛️🔥")

@st.cache_data
def optimize_image(image, max_size):
    """Redimensionnement optimisé avec cache"""
    img = ImageOps.exif_transpose(image)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def image_to_bytes(img_array, format='JPEG', quality=85):
    """Conversion avec gestion multi-format"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    if format == 'PNG':
        img.save(img_byte_arr, format=format, optimize=True)
    else:
        img.save(img_byte_arr, format=format, quality=quality, optimize=True)
    return img_byte_arr.getvalue()

def apply_distortion(img_rgb, intensity=0.5, frequency=10, mix=1.0):
    """Version optimisée avec pré-calcul des déformations"""
    rows, cols, channels = img_rgb.shape
    
    # Pré-calcul des déformations
    x = np.linspace(0, frequency * np.pi, cols)
    y = np.linspace(0, frequency * np.pi, rows)
    xx, yy = np.meshgrid(x, y)
    
    dx = intensity * np.sin(xx) * np.cos(yy) * 20
    dy = intensity * np.cos(xx) * np.sin(yy) * 20
    
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    new_x = np.clip(grid_x + dx, 0, cols-1)
    new_y = np.clip(grid_y + dy, 0, rows-1)
    
    coordinates = np.array([new_y.ravel(), new_x.ravel()])
    
    # Application sur chaque canal
    distorted_channels = []
    for channel in range(channels):
        channel_data = img_rgb[..., channel]
        distorted = ndimage.map_coordinates(channel_data, coordinates, order=1, mode='reflect')
        distorted = distorted.reshape(channel_data.shape)
        distorted_channels.append(channel_data * (1 - mix) + distorted * mix)
    
    return np.stack(distorted_channels, axis=-1)

def apply_sobel(image, mode='magnitude', boost=1.0, mix=1.0):
    """Nouvelle implémentation Sobel avec grayscale"""
    # Conversion en luminance
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    
    # Calcul des gradients
    if mode == 'magnitude':
        h = ndimage.sobel(gray, axis=1)
        v = ndimage.sobel(gray, axis=0)
        edges = np.sqrt(h**2 + v**2)
    elif mode == 'horizontal':
        edges = ndimage.sobel(gray, axis=1)
    elif mode == 'vertical':
        edges = ndimage.sobel(gray, axis=0)
    
    # Application sur tous les canaux
    edges_rgb = np.stack([edges]*3, axis=-1) * boost
    return image * (1 - mix) + np.clip(image + edges_rgb, 0, 1) * mix

# Contrôles latéraux
with st.sidebar:
    st.header("Contrôles FX")
    
    uploaded_file = st.file_uploader(
        f"Charger une image (max {MAX_FILE_SIZE_MB}MB)",
        type=["jpg", "png", "jpeg"],
        help="Les images trop grandes seront automatiquement redimensionnées"
    )
    
    if uploaded_file and len(uploaded_file.getvalue()) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        st.error("Fichier trop volumineux!")
        st.stop()
    
    filename_input = st.text_input("Nom du fichier", value="vmc_export")
    download_format = st.radio("Format de sortie", ['PNG', 'JPEG'], index=0)
    
    effects = st.multiselect(
        "Effets à appliquer",
        [
            'Sobel Magnitude', 
            'Sobel Horizontal', 
            'Sobel Vertical', 
            'Texture Analog', 
            'Décalage Chromatique',
            'Distortion',
            'Inversion des couleurs'
        ],
        default=['Sobel Magnitude']
    )
    
    params = {}
    if any('Sobel' in e for e in effects):
        params['sobel_boost'] = st.slider(
            "Intensité Sobel", 0.1, 5.0, 1.0, 0.1,
            help="Amplification des contours détectés"
        )
        params['sobel_mix'] = st.slider(
            "Mix Sobel", 0.0, 1.0, 1.0, 0.1,
            help="Mélange avec l'image originale"
        )
    
    if 'Texture Analog' in effects:
        params['grunge_intensity'] = st.slider(
            "Intensité Texture", 0.0, 1.0, 0.3,
            help="Intensité du bruit analogique"
        )
    
    if 'Décalage Chromatique' in effects:
        params['hue_shift'] = st.slider(
            "Décalage Hue", 0.0, 1.0, 0.0,
            help="Décalage de teinte colorimétrique"
        )
    
    if 'Distortion' in effects:
        params['distortion_intensity'] = st.slider(
            "Intensité Distortion", 0.0, 1.0, 0.5,
            help="Force de la distorsion"
        )
        params['distortion_freq'] = st.slider(
            "Fréquence Distortion", 1, 20, 8,
            help="Nombre de vagues par axe"
        )
        params['distortion_mix'] = st.slider(
            "Mix Distortion", 0.0, 1.0, 1.0,
            help="Mélange avec l'image non déformée"
        )
        
    if 'Inversion des couleurs' in effects:
        params['inversion_mix'] = st.slider(
            "Mix Inversion", 0.0, 1.0, 1.0,
            help="Mélange avec l'image originale"
        )
    
    params['global_mix'] = st.slider(
        "Mixage Global", 0.0, 1.0, 1.0, 0.1,
        help="Mélange final avec l'image originale"
    )

# Traitement principal
if uploaded_file and effects:
    try:
        with st.spinner("Chargement..."):
            img = Image.open(uploaded_file).convert('RGB')
            original_size = img.size
            
            if max(img.size) > MAX_IMAGE_SIZE:
                img = optimize_image(img, MAX_IMAGE_SIZE)
                st.warning(f"Redimensionné de {original_size} à {img.size}")

            img_array = np.array(img).astype(float)/255.0
            
            if max(img_array.shape[:2]) > PROCESSING_LIMIT:
                st.error("Image trop grande pour le traitement")
                st.stop()

        progress_bar = st.progress(0)
        result = np.copy(img_array)
        total_effects = len(effects)
        
        for idx, effect in enumerate(effects):
            progress_bar.progress((idx + 1) / total_effects)
            
            if 'Sobel' in effect:
                mode = effect.split()[-1].lower()
                result = apply_sobel(
                    result, 
                    mode=mode,
                    boost=params.get('sobel_boost', 1.0),
                    mix=params.get('sobel_mix', 1.0)
                )
            
            if effect == 'Texture Analog':
                noise = np.random.normal(0, params['grunge_intensity'], result.shape)
                result = np.clip(result + noise, 0, 1)
            
            if effect == 'Décalage Chromatique':
                hsv = mcolors.rgb_to_hsv(result)
                hsv[..., 0] = (hsv[..., 0] + params['hue_shift']) % 1.0
                result = mcolors.hsv_to_rgb(hsv)
            
            if effect == 'Distortion':
                result = apply_distortion(
                    result,
                    intensity=params['distortion_intensity'],
                    frequency=params['distortion_freq'],
                    mix=params['distortion_mix']
                )
            
            if effect == 'Inversion des couleurs':
                result = apply_inversion(result, params['inversion_mix'])

        final_output = np.clip(img_array * (1 - params['global_mix']) + result * params['global_mix'], 0, 1)
        progress_bar.empty()

        # Génération du fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_input}_{timestamp}.{download_format.lower()}"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(final_output, use_column_width=True, caption="SORTIE FINALE")
            
        with col2:
            st.download_button(
                "📥 Exporter",
                image_to_bytes(final_output, download_format, QUALITY),
                file_name=filename,
                mime=f"image/{download_format.lower()}"
            )
            
            st.markdown("**Analyse RGB:**")
            st.write(f"R: {final_output[...,0].mean():.2f} | G: {final_output[...,1].mean():.2f} | B: {final_output[...,2].mean():.2f}")

    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        st.stop()

else:
    st.info("⬅️ Chargez une image et sélectionnez des effets")

# Footer
st.markdown("---")
st.markdown("**VMC Collective** - Synthèse Ultimate v11.0")
