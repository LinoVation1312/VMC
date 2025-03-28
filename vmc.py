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

st.image(
    "https://m.media-amazon.com/images/M/MV5BYmZlOTY2OGUtYWY2Yy00NGE0LTg5YmQtNmM2MmYxOWI2YmJiXkEyXkFqcGdeQXVyMTkxNjUyNQ@@._V1_.jpg",
    width=500,
)

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
    
    x = np.linspace(0, frequency * np.pi, cols)
    y = np.linspace(0, frequency * np.pi, rows)
    xx, yy = np.meshgrid(x, y)
    
    dx = intensity * np.sin(xx) * np.cos(yy) * 20
    dy = intensity * np.cos(xx) * np.sin(yy) * 20
    
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    new_x = np.clip(grid_x + dx, 0, cols-1)
    new_y = np.clip(grid_y + dy, 0, rows-1)
    
    coordinates = np.array([new_y.ravel(), new_x.ravel()])
    
    distorted_channels = []
    for channel in range(channels):
        channel_data = img_rgb[..., channel]
        distorted = ndimage.map_coordinates(channel_data, coordinates, order=1, mode='reflect')
        distorted = distorted.reshape(channel_data.shape)
        distorted_channels.append(channel_data * (1 - mix) + distorted * mix)
    
    return np.stack(distorted_channels, axis=-1)

def spectral_diffraction(img, frequency=30, angle=0, dispersion=0.1, brightness=1.5):
    rows, cols = img.shape[:2]
    x = np.linspace(-np.pi, np.pi, cols)
    y = np.linspace(-np.pi, np.pi, rows)
    xx, yy = np.meshgrid(x, y)
    
    theta = np.radians(angle)
    xx_rot = xx * np.cos(theta) - yy * np.sin(theta)
    
    grating = np.sin(frequency * xx_rot)
    
    rgb_shift = [grating * dispersion * i for i in [1, 0.7, 0.4]]
    spectral = np.zeros_like(img)
    
    for i in range(3):
        shifted = np.roll(grating, int(rows * rgb_shift[i]), axis=1)
        spectral[..., i] = np.clip(shifted * brightness, 0, 1)
    
    return np.clip(img + spectral * brightness, 0, 1)

def analog_tape_distortion(img, saturation=0.8, noise_level=0.2, wow=0.1, flutter=0.05):
    hsv = mcolors.rgb_to_hsv(img)
    rows, cols = img.shape[:2]
    
    t = np.linspace(0, 20*np.pi, cols)
    wow_mod = np.sin(wow * t) * 0.1
    flutter_mod = np.sin(flutter * t * 50) * 0.05
    
    warp = np.zeros_like(img)
    for i in range(3):
        warp[..., i] = ndimage.shift(img[..., i], 
                                   (int(rows * wow_mod[i%cols]), 
                                    int(cols * flutter_mod[i%cols])),
                                   mode='wrap')
    
    compressed = np.arcsinh(img * saturation * 3) / 3
    noise = np.random.normal(0, noise_level, img.shape) * np.linspace(0.5, 1, cols)
    
    return np.clip(compressed + warp * 0.3 + noise, 0, 1)

def vinyl_texture(img, wear=0.5, dust=0.3, scratches=0.2, groove_depth=0.1):
    rows, cols = img.shape[:2]
    result = img.copy()
    
    y = np.linspace(-1, 1, rows)
    x = np.linspace(-1, 1, cols)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)
    grooves = (np.sin(radius * 200 * groove_depth) * 0.1 * wear)
    
    angles = np.random.rand(10) * 2 * np.pi
    scratch_pattern = sum(np.sin(10*radius + angle) for angle in angles)
    scratches = scratches * (scratch_pattern > 1.5).astype(float)
    
    dust_mask = np.random.rand(rows, cols) < dust * wear
    dust_effect = np.random.rand(*img.shape) * dust_mask[..., None]
    
    result = result * (1 - 0.3 * grooves[..., None]) 
    result = np.clip(result + dust_effect + scratches[..., None], 0, 1)
    result *= [1.0, 0.9, 0.8]
    
    return result

def holographic_effect(img, depth_map=None, iridescence=0.5, parallax=0.1):
    if depth_map is None:
        depth_map = np.sqrt(img[...,0]**2 + img[...,1]**2 + img[...,2]**2)
    
    rows, cols = img.shape[:2]
    x = np.linspace(0, 2*np.pi, cols)
    y = np.linspace(0, 2*np.pi, rows)
    xx, yy = np.meshgrid(x, y)
    
    interference = np.sin(10*xx + 5*yy) * np.cos(5*xx - 10*yy)
    
    rgb_shift = [interference * parallax * i for i in [1, 1.3, 1.6]]
    shifted = [ndimage.shift(img[...,i], (int(rows*s), int(cols*s)), mode='wrap') 
               for i, s in enumerate(rgb_shift)]
    
    hologram = np.stack(shifted, axis=-1)
    spectral_colors = np.sin(xx * yy * 50)[..., None] * np.array([0.3, 0.6, 1.0])
    
    return np.clip(img * (1 - iridescence) + hologram * depth_map[..., None] * iridescence + spectral_colors, 0, 1)

def apply_sobel(image, mode='magnitude', boost=1.0, mix=1.0):
    """Fonction corrigée avec retour de valeur"""
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    
    if mode == 'magnitude':
        h = ndimage.sobel(gray, axis=1)
        v = ndimage.sobel(gray, axis=0)
        edges = np.sqrt(h**2 + v**2)
    elif mode == 'horizontal':
        edges = ndimage.sobel(gray, axis=1)
    elif mode == 'vertical':
        edges = ndimage.sobel(gray, axis=0)
    
    edges_rgb = np.stack([edges]*3, axis=-1) * boost
    return image * (1 - mix) + np.clip(image + edges_rgb, 0, 1) * mix

def apply_inversion(img, mix=1.0):
    """Fonction séparée corrigée"""
    inverted = 1.0 - img
    return img * (1 - mix) + inverted * mix

# Contrôles latéraux
with st.sidebar:
    st.header("Contrôles FX")
    
    uploaded_file = st.file_uploader(
        f"Charger une image (max {MAX_FILE_SIZE_MB}MB)",
        type=["jpg", "png", "jpeg"]
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
            'Inversion des couleurs',
            'Diffraction Spectrale',
            'Distorsion Analogique', 
            'Texture Vinyle',     
            'Effet Holographique' 
        ],
        default=['Sobel Magnitude']
    )
    
    params = {}
    
    if 'Diffraction Spectrale' in effects:
        st.subheader("Paramètres Diffraction")
        params['diffraction_freq'] = st.slider("Fréquence réseau", 10, 100, 30)
        params['diffraction_disp'] = st.slider("Dispersion RGB", 0.0, 0.3, 0.1)
    
    if 'Distorsion Analogique' in effects:
        st.subheader("Paramètres Cassette")
        params['wow'] = st.slider("Fluctuation Wow", 0.0, 0.3, 0.1)
        params['flutter'] = st.slider("Fluctuation Flutter", 0.0, 0.2, 0.05)
    
    if 'Texture Vinyle' in effects:
        st.subheader("Paramètres Vinyle")
        params['grooves'] = st.slider("Profondeur sillons", 0.0, 0.5, 0.1)
        params['dust'] = st.slider("Poussière", 0.0, 1.0, 0.3)
    
    if 'Effet Holographique' in effects:
        st.subheader("Paramètres Hologramme")
        params['iridescence'] = st.slider("Iridescence", 0.0, 1.0, 0.5)
        params['parallax'] = st.slider("Parallaxe", 0.0, 0.3, 0.1)

    if any('Sobel' in e for e in effects):
        st.subheader("Paramètres Sobel")
        params['sobel_boost'] = st.slider("Intensité Sobel", 0.1, 5.0, 1.0, 0.1)
        params['sobel_mix'] = st.slider("Mix Sobel", 0.0, 1.0, 1.0, 0.1)
    
    if 'Texture Analog' in effects:
        st.subheader("Paramètres Texture")
        params['grunge_intensity'] = st.slider("Intensité Texture", 0.0, 1.0, 0.3)
    
    if 'Décalage Chromatique' in effects:
        st.subheader("Paramètres Chromatiques")
        params['hue_shift'] = st.slider("Décalage Hue", 0.0, 1.0, 0.0)
    
    if 'Distortion' in effects:
        st.subheader("Paramètres Distortion")
        params['distortion_intensity'] = st.slider("Intensité Distortion", 0.0, 1.0, 0.5)
        params['distortion_freq'] = st.slider("Fréquence Distortion", 1, 20, 8)
        params['distortion_mix'] = st.slider("Mix Distortion", 0.0, 1.0, 1.0)
        
    if 'Inversion des couleurs' in effects:
        st.subheader("Paramètres Inversion")
        params['inversion_mix'] = st.slider("Mix Inversion", 0.0, 1.0, 1.0)
    
    st.subheader("Paramètres Globaux")
    params['global_mix'] = st.slider("Mixage Global", 0.0, 1.0, 1.0, 0.1)

# Zone principale de traitement
if uploaded_file and effects:
    try:
        with st.spinner("Chargement..."):
            img = Image.open(uploaded_file).convert('RGB')
            original_size = img.size
            
            # Affichage original dans la zone principale
            st.image(img, caption="Image originale", use_column_width=True)
            
            if max(img.size) > MAX_IMAGE_SIZE:
                img = optimize_image(img, MAX_IMAGE_SIZE)
                st.warning(f"Redimensionné de {original_size} à {img.size}")

            img_array = np.array(img).astype(float)/255.0
            
            if img_array.size == 0:
                st.error("Erreur de conversion de l'image")
                st.stop()
            
            if max(img_array.shape[:2]) > PROCESSING_LIMIT:
                st.error("Image trop grande pour le traitement")
                st.stop()

        progress_bar = st.progress(0)
        result = np.copy(img_array)
        total_effects = len(effects)
        
        for idx, effect in enumerate(effects):
            progress_bar.progress((idx + 1) / total_effects)
            
            if effect == 'Sobel Magnitude':
                result = apply_sobel(result, mode='magnitude', boost=params.get('sobel_boost', 1.0), mix=params.get('sobel_mix', 1.0))
            elif effect == 'Sobel Horizontal':
                result = apply_sobel(result, mode='horizontal', boost=params.get('sobel_boost', 1.0), mix=params.get('sobel_mix', 1.0))
            elif effect == 'Sobel Vertical':
                result = apply_sobel(result, mode='vertical', boost=params.get('sobel_boost', 1.0), mix=params.get('sobel_mix', 1.0))
            elif effect == 'Texture Analog':
                noise = np.random.normal(0, params['grunge_intensity'], result.shape)
                result = np.clip(result + noise, 0, 1)
            elif effect == 'Décalage Chromatique':
                hsv = mcolors.rgb_to_hsv(result)
                hsv[..., 0] = (hsv[..., 0] + params['hue_shift']) % 1.0
                result = mcolors.hsv_to_rgb(hsv)
            elif effect == 'Distortion':
                result = apply_distortion(
                    result,
                    intensity=params['distortion_intensity'],
                    frequency=params['distortion_freq'],
                    mix=params['distortion_mix']
                )
            elif effect == 'Diffraction Spectrale':
                result = spectral_diffraction(
                    result,
                    frequency=params['diffraction_freq'],
                    dispersion=params['diffraction_disp']
                )
            elif effect == 'Distorsion Analogique':
                result = analog_tape_distortion(
                    result,
                    wow=params['wow'],
                    flutter=params['flutter']
                )
            elif effect == 'Texture Vinyle':
                result = vinyl_texture(
                    result,
                    groove_depth=params['grooves'],
                    dust=params['dust']
                )
            elif effect == 'Effet Holographique':
                result = holographic_effect(
                    result,
                    iridescence=params['iridescence'],
                    parallax=params['parallax']
                )
            elif effect == 'Inversion des couleurs':
                result = apply_inversion(result, params['inversion_mix'])

        final_output = np.clip(img_array * (1 - params['global_mix']) + result * params['global_mix'], 0, 1)
        
        if final_output is None:
            st.error("Aucun résultat à afficher")
            st.stop()

        progress_bar.empty()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_input}_{timestamp}.{download_format.lower()}"
    
        col1, col2 = st.columns([3, 1])
        with col1:
            if final_output.dtype != np.uint8:
                final_output = (final_output * 255).astype(np.uint8)
            st.image(final_output, use_container_width=True, caption="SORTIE FINALE")
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
