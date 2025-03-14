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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paramètres de performance
MAX_IMAGE_SIZE = 2000  # Taille max en pixels (côté le plus long)
MAX_FILE_SIZE_MB = 10  # Taille max du fichier en MB
QUALITY = 85           # Qualité JPEG (0-100)
PROCESSING_LIMIT = 4000  # Limite de traitement en pixels

# Style CSS
st.markdown(f"""
    <style>
    .main {{ background-color: #0E1117; color: #FAFAFA; }}
    .st-emotion-cache-6qob1r {{ background-color: #1A1D24 !important; }}
    h1 {{ color: #FF4B4B !important; font-family: 'Helvetica Neue', sans-serif; }}
    .warning {{ color: #FF4B4B !important; font-weight: bold; }}
    .info {{ color: #4BFF4B !important; }}
    </style>
    """, unsafe_allow_html=True)

# Header VMC
st.image("https://i.ibb.co/0jq6Y3N/vmc-logo.png", use_container_width=300)
st.title("VMC Ultimate FX Processor")
st.markdown("**Station de traitement visuel multi-effets** 🎛️🔥")

def optimize_image(image, max_size):
    """Redimensionne l'image tout en conservant le ratio"""
    img = ImageOps.exif_transpose(image)  # Corrige l'orientation
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def image_to_bytes(img_array, format='JPEG'):
    """Conversion avec compression optimisée"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format, quality=QUALITY, optimize=True)
    return img_byte_arr.getvalue()

def apply_distortion(img_rgb, intensity=0.5, frequency=10, mix=1.0):
    """Version corrigée pour les images RGB"""
    if len(img_rgb.shape) != 3:
        raise ValueError("L'image doit être au format RGB (3 canaux)")
    
    distorted_channels = []
    for channel in range(3):
        channel_data = img_rgb[..., channel]
        rows, cols = channel_data.shape
        
        # Génération des coordonnées
        x = np.linspace(0, frequency * np.pi, cols)
        y = np.linspace(0, frequency * np.pi, rows)
        xx, yy = np.meshgrid(x, y)
        
        # Calcul des déformations
        dx = intensity * np.sin(xx) * np.cos(yy) * 20
        dy = intensity * np.cos(xx) * np.sin(yy) * 20
        
        # Grille de coordonnées originale
        grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Nouvelles coordonnées
        new_x = np.clip(grid_x + dx, 0, cols-1)
        new_y = np.clip(grid_y + dy, 0, rows-1)
        
        # Application de la déformation
        coordinates = np.array([new_y.ravel(), new_x.ravel()])
        distorted = ndimage.map_coordinates(channel_data, coordinates, order=1, mode='reflect')
        distorted = distorted.reshape(channel_data.shape)
        
        # Mixage
        distorted_channel = channel_data * (1 - mix) + distorted * mix
        distorted_channels.append(distorted_channel)
    
    return np.stack(distorted_channels, axis=-1)

def apply_inversion(img, mix=1.0):
    return img * (1 - mix) + (1 - img) * mix

# Contrôles latéraux
with st.sidebar:
    st.header("Contrôles FX")
    
    # Gestion des fichiers uploadés
    uploaded_file = st.file_uploader(
        "Charger une image (max {}MB)".format(MAX_FILE_SIZE_MB),
        type=["jpg", "png", "jpeg"],
        help="Les images trop grandes seront automatiquement redimensionnées"
    )
    
    # Vérification de la taille du fichier
    if uploaded_file is not None:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Taille en MB
        if file_size > MAX_FILE_SIZE_MB:
            st.markdown(f'<p class="warning">Attention : Fichier trop volumineux ({file_size:.1f}MB > {MAX_FILE_SIZE_MB}MB)</p>', unsafe_allow_html=True)
            st.stop()
    
    filename_input = st.text_input("Nom du fichier", value="vmc_export")
    
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
    if any(e in effects for e in ['Sobel Magnitude', 'Sobel Horizontal', 'Sobel Vertical']):
        params['sobel_boost'] = st.slider("Intensité Sobel", 0.1, 5.0, 1.0, 0.1)
    
    if 'Texture Analog' in effects:
        params['grunge_intensity'] = st.slider("Intensité Texture", 0.0, 1.0, 0.3)
    
    if 'Décalage Chromatique' in effects:
        params['hue_shift'] = st.slider("Décalage Hue", 0.0, 1.0, 0.0)
    
    if 'Distortion' in effects:
        params['distortion_intensity'] = st.slider("Intensité Distortion", 0.0, 1.0, 0.5)
        params['distortion_freq'] = st.slider("Fréquence Distortion", 1, 20, 8)
        params['distortion_mix'] = st.slider("Mix Distortion", 0.0, 1.0, 1.0)
        
    if 'Inversion des couleurs' in effects:
        params['inversion_mix'] = st.slider("Mix Inversion", 0.0, 1.0, 1.0)
    
    params['global_mix'] = st.slider("Mixage Global", 0.0, 1.0, 1.0, 0.1)

# Traitement principal
if uploaded_file and effects:
    try:
        with st.spinner("Chargement et optimisation de l'image..."):
            # Chargement et optimisation
            img = Image.open(uploaded_file).convert('RGB')
            original_width, original_height = img.size
            
            # Redimensionnement si nécessaire
            if max(img.size) > MAX_IMAGE_SIZE:
                img = optimize_image(img, MAX_IMAGE_SIZE)
                st.markdown(f'<p class="warning">Image redimensionnée de {original_width}x{original_height} → {img.size[0]}x{img.size[1]}</p>', unsafe_allow_html=True)
            
            # Conversion en array numpy
            img_array = np.array(img).astype(float)/255.0
            result = np.copy(img_array)
            
            # Vérification de la taille pour le traitement
            if max(img_array.shape[:2]) > PROCESSING_LIMIT:
                st.error(f"L'image est trop grande pour le traitement (max {PROCESSING_LIMIT}px)")
                st.stop()
            
            # Application des effets
            for effect in effects:
                if 'Sobel' in effect:
                    mode = effect.split()[-1].lower()
                    h = ndimage.sobel(result[..., 0], axis=0 if 'horizontal' in mode else 1)
                    v = ndimage.sobel(result[..., 1], axis=0 if 'horizontal' in mode else 1)
                    edges = np.stack([h, v, np.zeros_like(h)], axis=-1) * params['sobel_boost']
                    result = np.clip(result + edges, 0, 1)
                
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

            # Mixage final
            final_output = np.clip(img_array * (1 - params['global_mix']) + result * params['global_mix'], 0, 1)

        # Génération du nom de fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_input}_{timestamp}.png"

        # Affichage
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(final_output, use_container_width=True, caption="SORTIE FINALE")
            
        with col2:
            st.download_button(
                "📥 Exporter",
                image_to_bytes(final_output, 'PNG'),
                file_name=filename,
                mime="image/png"
            )
            
            st.markdown("**Analyse RGB:**")
            rgb_mean = final_output.mean(axis=(0,1))
            st.write(f"R: {rgb_mean[0]:.2f} | G: {rgb_mean[1]:.2f} | B: {rgb_mean[2]:.2f}")

    except Exception as e:
        st.error(f"Erreur de traitement : {str(e)}")
        st.stop()

else:
    st.info("⬅️ Chargez une image et sélectionnez des effets")

# Footer
st.markdown("---")
st.markdown("**VMC Collective** - Synthèse Ultimate v10.0")
