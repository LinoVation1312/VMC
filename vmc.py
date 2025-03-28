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
MAX_IMAGE_SIZE =2000
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

# Correction pour la diffraction spectrale
def spectral_diffraction(img, frequency=30, angle=0, dispersion=0.3, brightness=2.0):
    rows, cols = img.shape[:2]
    x = np.linspace(-np.pi*4, np.pi*4, cols)
    y = np.linspace(-np.pi*4, np.pi*4, rows)
    xx, yy = np.meshgrid(x, y)
    
    # Motif d'interférence complexe
    grating = np.sin(frequency * xx) * np.cos(frequency * yy * 0.7)
    grating += 0.5 * np.sin(0.7 * frequency * (xx * np.cos(np.radians(45)) + yy * np.sin(np.radians(45))))
    
    # Décalages chromatiques différentiels
    shifts = [
        (int(rows * dispersion * 0.8), int(cols * dispersion * 0.2)),
        (int(rows * dispersion * 0.4), int(cols * dispersion * 0.6)),
        (int(rows * dispersion * 0.1), int(cols * dispersion * 0.9))
    ]
    
    spectral = np.zeros_like(img)
    for i in range(3):
        shifted = np.roll(grating, shifts[i][0], axis=0)
        shifted = np.roll(shifted, shifts[i][1], axis=1)
        spectral[..., i] = np.clip(np.abs(shifted) * brightness * (i+1)/3, 0, 1)
    
    # Combinaison non linéaire
    return np.clip(img * 0.7 + spectral * 1.3 - 0.3, 0, 1)


def analog_tape_distortion(img, saturation=1.5, noise_level=0.4, 
                          wow=0.4, flutter=0.3, offset_mix=0.5, flutter_boost=2.0):
    """
    Nouveaux paramètres :
    - offset_mix: Contrôle le décalage entre les deux couches déformées (0-1)
    - flutter_boost: Amplification supplémentaire de l'effet flutter (1-5)
    """
    rows, cols = img.shape[:2]
    t = np.linspace(0, 6*np.pi, cols)
    
    # Génération de motifs complexes avec harmoniques
    wow_mod = 0.15 * (np.sin(wow * t) + 0.5 * np.sin(2.7 * wow * t))
    flutter_mod = flutter_boost * 0.2 * (np.sin(flutter * t * 50) + 
                                       0.3 * np.cos(flutter * t * 27 * np.pi))
    
    # Création de deux couches déformées différentes
    warp_layer1 = np.zeros_like(img)
    warp_layer2 = np.zeros_like(img)
    
    for i in range(3):
        # Première couche avec déformation principale
        warp_layer1[..., i] = ndimage.shift(
            img[..., i],
            (int(rows * wow_mod[i%cols] * 2), 
            int(cols * flutter_mod[i%cols] * 1.5),
            mode='reflect', order=3
        )
        
        # Deuxième couche avec décalage alternatif
        warp_layer2[..., i] = ndimage.shift(
            img[..., i],
            (int(rows * wow_mod[i%cols] * 1.3), 
            int(cols * flutter_mod[i%cols] * 2 * (-1 if i==1 else 1)),
            mode='mirror', order=2
        )
    
    # Mélange des deux couches déformées
    warped = warp_layer1 * (1 - offset_mix) + warp_layer2 * offset_mix
    
    # Compression non-linéaire accentuée
    compressed = np.tanh(img * saturation * 3) * 0.9
    
    # Bruit directionnel amélioré
    noise_profile = np.linspace(0.3, 1, cols)[None, :, None] * np.linspace(0.8, 1.2, rows)[:, None, None]
    noise = np.random.normal(0, noise_level**1.5, img.shape) * noise_profile
    
    # Combinaison finale plus dynamique
    return np.clip(
        compressed * 0.6 + 
        warped * 0.8 + 
        noise * 1.2 -
        0.2 * (compressed * warped),
        0, 1
    )

def vinyl_texture(img, wear=0.5, dust=0.3, scratches=0.2, groove_depth=0.15):
    rows, cols = img.shape[:2]
    y = np.linspace(-8, 8, rows)
    x = np.linspace(-8, 8, cols)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2) * (1 + 0.1 * np.sin(yy * 75))  # Spirale serrée
    
    # Micro-sillons (100 lignes/mm)
    grooves = (np.sin(radius * 150 + yy * 30) * 0.08 * groove_depth)/(groove_depth**2+1e-9)
    grooves += 0.03 * np.sin(50 * radius) * groove_depth
    
    # Rayures microscopiques aléatoires
    scratch_pattern = sum(0.2 * np.sin(100*radius + np.random.rand()*10) for _ in range(20))
    scratches = (scratch_pattern > 0.9).astype(float) * scratches
    
    # Texture de surface
    texture = np.random.rand(rows, cols) * 0.1 * wear
    result = np.clip(img * (0.9 + 0.1 * grooves[..., None]) + texture[..., None] + scratches[..., None], 0, 1)
    
    return result * [0.91, 0.90, 0.85]  # Teinte neutre


# Correction pour l'effet holographique
def holographic_effect(img, depth_map=None, iridescence=0.7, parallax=0.2):
    rows, cols = img.shape[:2]
    
    # Génération de franges d'interférence
    x = np.linspace(0, 6*np.pi, cols)
    y = np.linspace(0, 6*np.pi, rows)
    xx, yy = np.meshgrid(x, y)
    interference = np.sin(8*xx) * np.cos(6*yy) + 0.5*np.sin(3*(xx + yy))
    
    # Décalages chromatiques dynamiques
    shifts = [
        (int(rows * parallax * 0.2), 
        int(rows * parallax * 0.3), 
        int(rows * parallax * 0.4))
    ]
    
    shifted = [
        ndimage.shift(img[...,i], (shifts[i], shifts[i]), mode='wrap')
        for i in range(3)
    ]
    hologram = np.stack(shifted, axis=-1)
    
    # Couleurs spectrales directionnelles
    angle_map = np.arctan2(yy - rows/2, xx - cols/2)
    spectral = np.stack([
        np.cos(angle_map * 3),
        np.sin(angle_map * 3 + np.pi/3),
        np.cos(angle_map * 3 - np.pi/3)
    ], axis=-1) * 0.4
    
    return np.clip(
        img * (1 - iridescence) + 
        hologram * iridescence * 0.8 + 
        spectral * iridescence * 0.6, 
        0, 1
    )
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
            st.image(img, caption="Image originale", width=400)
            
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
