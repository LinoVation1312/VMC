import numpy as np
import streamlit as st
from scipy import ndimage
from PIL import Image, ImageOps
import io
import matplotlib.colors as mcolors
from datetime import datetime

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VMC Poster FX",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded"
)

MAX_IMAGE_SIZE = 2000
MAX_FILE_SIZE_MB = 10
QUALITY = 92
PROCESSING_LIMIT = 4000

# ─── STYLE ──────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&display=swap');
    .main { background-color: #0A0A0A; color: #F0EDE6; }
    h1 { font-family: 'Bebas Neue', sans-serif !important; 
         font-size: 3rem !important; 
         letter-spacing: 4px !important;
         color: #F5E642 !important; }
    h2, h3 { font-family: 'Space Mono', monospace !important; color: #F5E642 !important; }
    .st-emotion-cache-6qob1r { background-color: #111111 !important; border-right: 1px solid #222; }
    .stProgress > div > div > div > div { background-color: #F5E642 !important; }
    .stDownloadButton > button { 
        background-color: #F5E642 !important; 
        color: #0A0A0A !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("VMC POSTER FX")
st.markdown("**Station de traitement visuel pour affiches & créations graphiques**")

# ─── UTILS ──────────────────────────────────────────────────────────────────────
@st.cache_data
def optimize_image(image, max_size):
    img = ImageOps.exif_transpose(image)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def to_float(img_array):
    if img_array.dtype == np.uint8:
        return img_array.astype(float) / 255.0
    return img_array.astype(float)

def image_to_bytes(img_array, fmt='JPEG', quality=92):
    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    if fmt == 'PNG':
        img.save(buf, format='PNG', optimize=True)
    else:
        img.save(buf, format='JPEG', quality=quality, optimize=True)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
#  EFFETS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_sobel(img, mode='magnitude', boost=1.0, mix=1.0):
    """Détection de contours Sobel — dessine les arêtes de l'image."""
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    if mode == 'magnitude':
        edges = np.sqrt(ndimage.sobel(gray, axis=1)**2 + ndimage.sobel(gray, axis=0)**2)
    elif mode == 'horizontal':
        edges = np.abs(ndimage.sobel(gray, axis=1))
    else:
        edges = np.abs(ndimage.sobel(gray, axis=0))
    edges_rgb = np.stack([edges] * 3, axis=-1) * boost
    return np.clip(img * (1 - mix) + np.clip(img + edges_rgb, 0, 1) * mix, 0, 1)


def apply_inversion(img, mix=1.0):
    """Inversion des valeurs — négatif photographique."""
    return np.clip(img * (1 - mix) + (1.0 - img) * mix, 0, 1)


def apply_hue_shift(img, shift=0.0):
    """Rotation de la teinte dans l'espace HSV."""
    hsv = mcolors.rgb_to_hsv(np.clip(img, 0, 1))
    hsv[..., 0] = (hsv[..., 0] + shift) % 1.0
    return np.clip(mcolors.hsv_to_rgb(hsv), 0, 1)


def apply_distortion(img, intensity=0.5, frequency=10, mix=1.0):
    """Déformation ondulatoire sinusoïdale — effet liquide."""
    rows, cols, channels = img.shape
    x = np.linspace(0, frequency * np.pi, cols)
    y = np.linspace(0, frequency * np.pi, rows)
    xx, yy = np.meshgrid(x, y)
    dx = intensity * np.sin(xx) * np.cos(yy) * 20
    dy = intensity * np.cos(xx) * np.sin(yy) * 20
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    new_x = np.clip(gx + dx, 0, cols - 1)
    new_y = np.clip(gy + dy, 0, rows - 1)
    coords = np.array([new_y.ravel(), new_x.ravel()])
    distorted = np.stack([
        ndimage.map_coordinates(img[..., c], coords, order=1, mode='reflect').reshape(rows, cols)
        for c in range(channels)
    ], axis=-1)
    return np.clip(img * (1 - mix) + distorted * mix, 0, 1)


def holographic_effect(img, iridescence=0.5, parallax=0.1):
    """Interférence lumineuse — reflet holographique multi-couches."""
    rows, cols = img.shape[:2]
    xx, yy = np.meshgrid(
        np.linspace(0, 2 * np.pi, cols),
        np.linspace(0, 2 * np.pi, rows)
    )
    shifts = [int(rows * parallax * k) for k in (0.08, 0.13, 0.18)]
    shifted = np.stack([
        ndimage.shift(img[..., i], (shifts[i], shifts[i]), mode='wrap')
        for i in range(3)
    ], axis=-1)
    spectral = np.sin(xx * yy * 50)[..., None] * np.array([0.3, 0.6, 1.0])
    depth = np.sqrt(np.sum(img ** 2, axis=-1, keepdims=True))
    return np.clip(
        img * (1 - iridescence) +
        shifted * depth * iridescence * 0.7 +
        spectral * iridescence * 0.5,
        0, 1
    )


def neon_effect(img, hue=0.8, intensity=1.5, mix=0.7, saturation=0.9,
                glow_spread=2.0, thin_spread=0.6, bg_darken=0.3):
    """
    Néon — contours fins et lumineux sur fond sombre.
    Utilise le laplacien pour des lignes nettes, puis trois couches de bloom
    (fin / moyen / large) pour simuler la diffusion lumineuse du verre.
    """
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    # Laplacien : contours très fins, bi-directionnel
    lap = np.abs(ndimage.laplace(gray))
    # Normalisation robuste
    p99 = np.percentile(lap, 99)
    if p99 > 0:
        lap = np.clip(lap / p99, 0, 1)

    # Trois couches de bloom avec sigma croissants
    core   = ndimage.gaussian_filter(lap, sigma=max(0.4, thin_spread * 0.5))
    mid    = ndimage.gaussian_filter(lap, sigma=glow_spread)
    halo   = ndimage.gaussian_filter(lap, sigma=glow_spread * 3.5) * 0.4

    # Combinaison : core très bright, halo ambiant
    glow = np.clip(core * 1.8 + mid * 1.0 + halo, 0, 1) * intensity
    glow = np.clip(glow, 0, 1)

    # Couleur : teinte fixe + légère variation spatiale via position Y
    rows, cols = img.shape[:2]
    hue_map = (hue + np.linspace(0, 0.04, rows)[:, None]) % 1.0
    hue_map = np.broadcast_to(hue_map[..., None], (rows, cols, 1))

    hsv = np.zeros((rows, cols, 3))
    hsv[..., 0] = hue_map[..., 0]
    hsv[..., 1] = saturation
    hsv[..., 2] = glow
    neon_rgb = mcolors.hsv_to_rgb(hsv)

    # Fond assombri pour faire ressortir l'effet (comme une enseigne sur mur noir)
    dark_bg = img * (1.0 - bg_darken * glow[..., None])

    return np.clip(dark_bg * (1 - mix) + np.clip(dark_bg + neon_rgb, 0, 1) * mix, 0, 1)


# ── NOUVEAUX EFFETS ─────────────────────────────────────────────────────────────

def duotone(img, color_shadows=(0.05, 0.0, 0.2), color_highlights=(1.0, 0.9, 0.1), mix=1.0):
    """
    Duotone (style Spotify) — mappe les ombres et les hautes lumières sur deux couleurs.
    Idéal pour affiches haut-contraste.
    """
    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])[..., None]
    shadows = np.array(color_shadows)
    highlights = np.array(color_highlights)
    result = gray * highlights + (1 - gray) * shadows
    return np.clip(img * (1 - mix) + result * mix, 0, 1)


def glitch_rgb(img, shift_h=8, shift_v=3, num_bands=6, intensity=1.0):
    """
    Glitch RGB — décalage de canaux couleur par bandes horizontales aléatoires.
    Rendu numérique brutaliste.
    """
    result = img.copy()
    rows, cols = img.shape[:2]
    rng = np.random.default_rng(42)
    band_positions = rng.integers(0, rows, num_bands)
    band_heights = rng.integers(5, max(6, rows // 8), num_bands)

    for pos, height in zip(band_positions, band_heights):
        y0 = int(pos)
        y1 = min(rows, y0 + int(height))
        sx = int(rng.integers(-shift_h, shift_h + 1) * intensity)
        sy = int(rng.integers(-shift_v, shift_v + 1) * intensity)

        # canal R décalé horizontalement
        result[y0:y1, :, 0] = np.roll(img[y0:y1, :, 0], sx, axis=1)
        # canal B décalé dans l'autre sens + vertical
        shifted_b = np.roll(img[y0:y1, :, 2], -sx, axis=1)
        if sy != 0:
            y0b = max(0, y0 + sy)
            y1b = min(rows, y1 + sy)
            h = min(y1b - y0b, y1 - y0)
            if h > 0:
                result[y0b:y0b + h, :, 2] = shifted_b[:h]
        else:
            result[y0:y1, :, 2] = shifted_b

    return np.clip(result, 0, 1)


def halftone(img, dot_size=6, angle=45, style='dots', mix=1.0):
    """
    Trame halftone — simulation offset/sérigraphie.
    Styles : dots (points), lines (lignes), diamonds (losanges).
    """
    rows, cols = img.shape[:2]
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    rad = np.radians(angle)
    xs = np.arange(cols)
    ys = np.arange(rows)
    xx, yy = np.meshgrid(xs, ys)
    xr = xx * np.cos(rad) + yy * np.sin(rad)
    yr = -xx * np.sin(rad) + yy * np.cos(rad)

    xmod = (xr % dot_size) / dot_size - 0.5
    ymod = (yr % dot_size) / dot_size - 0.5

    if style == 'dots':
        pattern = np.sqrt(xmod ** 2 + ymod ** 2)
        threshold = (1 - gray) * 0.7 * 0.5
        halftone_mask = (pattern < threshold).astype(float)
    elif style == 'lines':
        pattern = np.abs(ymod)
        threshold = (1 - gray) * 0.5
        halftone_mask = (pattern < threshold).astype(float)
    else:  # diamonds
        pattern = np.abs(xmod) + np.abs(ymod)
        threshold = (1 - gray) * 0.7 * 0.5
        halftone_mask = (pattern < threshold).astype(float)

    halftone_rgb = np.stack([halftone_mask] * 3, axis=-1)
    return np.clip(img * (1 - mix) + halftone_rgb * mix, 0, 1)


def poster_solarize(img, levels=4, mix=1.0):
    """
    Postérisation / Solarisation — réduction à N niveaux de couleur façon Warhol.
    Transforme une photo en affiche graphique impactante.
    """
    quantized = np.round(img * (levels - 1)) / (levels - 1)
    return np.clip(img * (1 - mix) + quantized * mix, 0, 1)


def solarize(img, threshold=0.5, invert_above=True, channel_thresholds=None, mix=1.0):
    """
    Solarisation Man Ray — inversion des valeurs dépassant un seuil lumineux.
    Au-delà du seuil : les tons clairs basculent dans leur négatif.
    En dessous : l'image reste intacte.
    channel_thresholds : tuple (r, g, b) pour seuils indépendants par canal,
    sinon utilise `threshold` pour les 3.
    """
    result = img.copy()
    if channel_thresholds is not None:
        for c, thr in enumerate(channel_thresholds):
            ch = img[..., c]
            mask = ch > thr if invert_above else ch < thr
            result[..., c] = np.where(mask, 1.0 - ch, ch)
    else:
        mask = img > threshold if invert_above else img < threshold
        result = np.where(mask, 1.0 - img, img)
    return np.clip(img * (1 - mix) + result * mix, 0, 1)


def vignette_grain(img, vignette_strength=0.6, grain_amount=0.08, grain_size=1, mix=1.0):
    """
    Vignette + Grain — finition photo argentique.
    Assombrit les bords et ajoute une texture de grain.
    """
    rows, cols = img.shape[:2]
    xs, ys = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    vignette = 1 - np.clip(np.sqrt(xs ** 2 + ys ** 2) * vignette_strength, 0, 1)
    result = img * vignette[..., None]
    rng = np.random.default_rng(7)
    grain = rng.normal(0, grain_amount, img.shape)
    if grain_size > 1:
        grain = ndimage.uniform_filter(grain, size=grain_size)
    result = np.clip(result + grain, 0, 1)
    return np.clip(img * (1 - mix) + result * mix, 0, 1)


def chromatic_aberration(img, strength=8, falloff=0.6, mix=1.0):
    """
    Aberration Chromatique — dispersion des canaux R/B depuis le centre.
    Simule un objectif imparfait : effet photo/ciné très recherché.
    """
    rows, cols = img.shape[:2]
    xs, ys = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    dist = np.sqrt(xs ** 2 + ys ** 2)
    weight = dist ** falloff

    result = img.copy()
    for shift, channel in [(strength, 0), (-strength, 2)]:
        dx = (xs * weight * shift).astype(int)
        dy = (ys * weight * shift).astype(int)
        src_x = np.clip(np.arange(cols) + dx, 0, cols - 1)
        src_y = np.clip(np.arange(rows)[:, None] + dy, 0, rows - 1)
        result[..., channel] = img[src_y, src_x, channel]

    return np.clip(img * (1 - mix) + result * mix, 0, 1)


def spectral_diffraction(img, frequency=30, dispersion=0.15, brightness=1.5, mix=1.0):
    """
    Diffraction Spectrale — réseau de diffraction prismatique.
    Ajoute des reflets arc-en-ciel sur les zones lumineuses.
    """
    rows, cols = img.shape[:2]
    xx, yy = np.meshgrid(
        np.linspace(-np.pi * 4, np.pi * 4, cols),
        np.linspace(-np.pi * 4, np.pi * 4, rows)
    )
    grating = np.sin(frequency * xx) * np.cos(frequency * yy * 0.7)
    grating += 0.5 * np.sin(0.7 * frequency * (xx * 0.707 + yy * 0.707))

    shifts = [
        (int(rows * dispersion * k1), int(cols * dispersion * k2))
        for k1, k2 in [(0.8, 0.2), (0.4, 0.6), (0.1, 0.9)]
    ]
    spectral = np.zeros_like(img)
    for i in range(3):
        shifted = np.roll(np.roll(grating, shifts[i][0], axis=0), shifts[i][1], axis=1)
        spectral[..., i] = np.clip(np.abs(shifted) * brightness * (i + 1) / 3, 0, 1)

    result = np.clip(img * 0.7 + spectral * 1.3 - 0.2, 0, 1)
    return np.clip(img * (1 - mix) + result * mix, 0, 1)


def cross_process(img, mix=1.0):
    """
    Cross-Processing — simulation du développement inversé E-6/C-41.
    Virages couleur contrastés : verts froids dans les ombres, jaunes chauds dans les hautes lumières.
    """
    r = np.clip(img[..., 0] ** 0.7 * 1.3, 0, 1)
    g = np.clip(img[..., 1] ** 1.2 * 0.9 + 0.05, 0, 1)
    b = np.clip(img[..., 2] ** 1.5 * 0.7 + img[..., 2] * 0.1, 0, 1)
    result = np.stack([r, g, b], axis=-1)
    return np.clip(img * (1 - mix) + result * mix, 0, 1)


# ─── ALL EFFECTS REGISTRY ───────────────────────────────────────────────────────
EFFECTS_LIST = [
    # Contours & formes
    "Sobel (magnitude)",
    "Sobel (horizontal)",
    "Sobel (vertical)",
    # Couleurs
    "Décalage de teinte",
    "Inversion",
    "Postérisation Warhol",
    "Cross-Process",
    "Duotone",
    # Optique & distorsion
    "Distorsion liquide",
    "Aberration Chromatique",
    "Diffraction Spectrale",
    "Holographique",
    # Lumière
    "Néon",
    "Solarisation",
    # Texture & finition
    "Trame Halftone",
    "Glitch RGB",
    "Vignette + Grain",
]

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙ CONTRÔLES")

    uploaded_file = st.file_uploader(
        f"Image source (max {MAX_FILE_SIZE_MB}MB)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file and len(uploaded_file.getvalue()) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        st.error("Fichier trop volumineux!")
        st.stop()

    filename_input = st.text_input("Nom du fichier export", value="vmc_poster")
    download_format = st.radio("Format", ['PNG', 'JPEG'], index=0)

    effects = st.multiselect(
        "Effets (appliqués dans l'ordre)",
        EFFECTS_LIST,
        default=["Duotone"]
    )

    params = {}

    # ── Sobel
    if any("Sobel" in e for e in effects):
        with st.expander("Sobel"):
            params['sobel_boost'] = st.slider("Intensité", 0.1, 5.0, 1.2, 0.1, key="sobel_boost")
            params['sobel_mix']   = st.slider("Mix", 0.0, 1.0, 1.0, 0.05, key="sobel_mix")

    # ── Décalage teinte
    if "Décalage de teinte" in effects:
        with st.expander("Décalage de teinte"):
            params['hue_shift'] = st.slider("Rotation teinte", 0.0, 1.0, 0.15, 0.01, key="hue_shift")

    # ── Inversion
    if "Inversion" in effects:
        with st.expander("Inversion"):
            params['inversion_mix'] = st.slider("Mix", 0.0, 1.0, 1.0, 0.05, key="inversion_mix")

    # ── Postérisation
    if "Postérisation Warhol" in effects:
        with st.expander("Postérisation"):
            params['poster_levels'] = st.slider("Niveaux de couleur", 2, 8, 4, key="poster_levels")
            params['poster_mix']    = st.slider("Mix", 0.0, 1.0, 1.0, 0.05, key="poster_mix")

    # ── Cross-Process
    if "Cross-Process" in effects:
        with st.expander("Cross-Process"):
            params['cross_mix'] = st.slider("Mix", 0.0, 1.0, 0.9, 0.05, key="cross_mix")

    # ── Duotone
    if "Duotone" in effects:
        with st.expander("Duotone"):
            st.markdown("**Couleur ombres (RGB 0-255)**")
            sr = st.slider("Ombre R", 0, 255, 10, key="duo_sr")
            sg = st.slider("Ombre G", 0, 255, 0,  key="duo_sg")
            sb = st.slider("Ombre B", 0, 255, 60, key="duo_sb")
            st.markdown("**Couleur hautes lumières**")
            hr = st.slider("Lumière R", 0, 255, 255, key="duo_hr")
            hg = st.slider("Lumière G", 0, 255, 230, key="duo_hg")
            hb = st.slider("Lumière B", 0, 255, 20,  key="duo_hb")
            params['duo_shadows']    = (sr / 255, sg / 255, sb / 255)
            params['duo_highlights'] = (hr / 255, hg / 255, hb / 255)
            params['duo_mix']        = st.slider("Mix", 0.0, 1.0, 1.0, 0.05, key="duo_mix")

    # ── Distorsion liquide
    if "Distorsion liquide" in effects:
        with st.expander("Distorsion liquide"):
            params['dist_intensity'] = st.slider("Intensité", 0.0, 1.0, 0.4, 0.05, key="dist_int")
            params['dist_freq']      = st.slider("Fréquence", 1, 20, 6, key="dist_freq")
            params['dist_mix']       = st.slider("Mix", 0.0, 1.0, 0.8, 0.05, key="dist_mix")

    # ── Aberration Chromatique
    if "Aberration Chromatique" in effects:
        with st.expander("Aberration Chromatique"):
            params['ca_strength'] = st.slider("Force", 1, 30, 10, key="ca_strength")
            params['ca_falloff']  = st.slider("Falloff radial", 0.2, 2.0, 0.6, 0.1, key="ca_falloff")
            params['ca_mix']      = st.slider("Mix", 0.0, 1.0, 1.0, 0.05, key="ca_mix")

    # ── Diffraction
    if "Diffraction Spectrale" in effects:
        with st.expander("Diffraction Spectrale"):
            params['diff_freq'] = st.slider("Fréquence réseau", 10, 100, 30, key="diff_freq")
            params['diff_disp'] = st.slider("Dispersion", 0.0, 0.3, 0.12, 0.01, key="diff_disp")
            params['diff_mix']  = st.slider("Mix", 0.0, 1.0, 0.8, 0.05, key="diff_mix")

    # ── Holographique
    if "Holographique" in effects:
        with st.expander("Holographique"):
            params['holo_irid'] = st.slider("Iridescence", 0.0, 1.0, 0.5, 0.05, key="holo_irid")
            params['holo_par']  = st.slider("Parallaxe", 0.0, 0.3, 0.1, 0.01, key="holo_par")

    # ── Néon
    if "Néon" in effects:
        with st.expander("Néon"):
            params['neon_hue']    = st.slider("Teinte", 0.0, 1.0, 0.75, 0.01, key="neon_hue")
            params['neon_sat']    = st.slider("Saturation", 0.0, 1.0, 0.95, 0.05, key="neon_sat")
            params['neon_int']    = st.slider("Intensité", 0.1, 4.0, 2.0, 0.1, key="neon_int")
            params['neon_spread'] = st.slider("Bloom spread", 0.3, 8.0, 2.5, 0.1, key="neon_spread")
            params['neon_thin']   = st.slider("Finesse core", 0.1, 3.0, 0.6, 0.1, key="neon_thin")
            params['neon_dark']   = st.slider("Assombrissement fond", 0.0, 1.0, 0.35, 0.05, key="neon_dark")
            params['neon_mix']    = st.slider("Mix", 0.0, 1.0, 0.85, 0.05, key="neon_mix")

    # ── Solarisation
    if "Solarisation" in effects:
        with st.expander("Solarisation"):
            sol_mode = st.radio("Mode seuil", ["Global", "Par canal"], key="sol_mode")
            if sol_mode == "Global":
                params['sol_threshold'] = st.slider("Seuil", 0.0, 1.0, 0.5, 0.01, key="sol_thr")
                params['sol_channels'] = None
            else:
                tr = st.slider("Seuil R", 0.0, 1.0, 0.45, 0.01, key="sol_tr")
                tg = st.slider("Seuil G", 0.0, 1.0, 0.50, 0.01, key="sol_tg")
                tb = st.slider("Seuil B", 0.0, 1.0, 0.55, 0.01, key="sol_tb")
                params['sol_threshold'] = 0.5
                params['sol_channels'] = (tr, tg, tb)
            params['sol_invert_above'] = st.checkbox("Inverser au-dessus du seuil", True, key="sol_inv")
            params['sol_mix'] = st.slider("Mix", 0.0, 1.0, 1.0, 0.05, key="sol_mix")

    # ── Halftone
    if "Trame Halftone" in effects:
        with st.expander("Trame Halftone"):
            params['half_size']  = st.slider("Taille des points", 1, 20, 6, key="half_size")
            params['half_angle'] = st.slider("Angle (°)", 0, 90, 45, key="half_angle")
            params['half_style'] = st.selectbox("Style", ['dots', 'lines', 'diamonds'], key="half_style")
            params['half_mix']   = st.slider("Mix", 0.0, 1.0, 1.0, 0.05, key="half_mix")

    # ── Glitch
    if "Glitch RGB" in effects:
        with st.expander("Glitch RGB"):
            params['glitch_h']     = st.slider("Décalage horizontal", 1, 40, 12, key="glitch_h")
            params['glitch_v']     = st.slider("Décalage vertical", 0, 20, 4, key="glitch_v")
            params['glitch_bands'] = st.slider("Nombre de bandes", 2, 20, 8, key="glitch_bands")
            params['glitch_int']   = st.slider("Intensité", 0.1, 2.0, 1.0, 0.1, key="glitch_int")

    # ── Vignette Grain
    if "Vignette + Grain" in effects:
        with st.expander("Vignette + Grain"):
            params['vig_str']   = st.slider("Force vignette", 0.0, 1.5, 0.6, 0.05, key="vig_str")
            params['grain_amt'] = st.slider("Quantité grain", 0.0, 0.25, 0.06, 0.01, key="grain_amt")
            params['grain_sz']  = st.slider("Taille grain", 1, 5, 1, key="grain_sz")
            params['vg_mix']    = st.slider("Mix", 0.0, 1.0, 1.0, 0.05, key="vg_mix")

    st.markdown("---")
    st.subheader("Global")
    params['global_mix'] = st.slider("Mixage Global", 0.0, 1.0, 1.0, 0.05, key="global_mix")


# ─── MAIN PROCESSING ────────────────────────────────────────────────────────────
if uploaded_file and effects:
    try:
        with st.spinner("Chargement de l'image..."):
            img_pil = Image.open(uploaded_file).convert('RGB')
            original_size = img_pil.size
            if max(img_pil.size) > MAX_IMAGE_SIZE:
                img_pil = optimize_image(img_pil, MAX_IMAGE_SIZE)
                st.warning(f"Redimensionné : {original_size} → {img_pil.size}")
            st.image(img_pil, caption="Image source", width=420)
            img_array = np.array(img_pil).astype(float) / 255.0
            if max(img_array.shape[:2]) > PROCESSING_LIMIT:
                st.error("Image trop grande pour le traitement.")
                st.stop()

        progress = st.progress(0)
        result = img_array.copy()

        DISPATCH = {
            "Sobel (magnitude)":    lambda r: apply_sobel(r, 'magnitude', params.get('sobel_boost', 1.0), params.get('sobel_mix', 1.0)),
            "Sobel (horizontal)":   lambda r: apply_sobel(r, 'horizontal', params.get('sobel_boost', 1.0), params.get('sobel_mix', 1.0)),
            "Sobel (vertical)":     lambda r: apply_sobel(r, 'vertical', params.get('sobel_boost', 1.0), params.get('sobel_mix', 1.0)),
            "Décalage de teinte":   lambda r: apply_hue_shift(r, params.get('hue_shift', 0.15)),
            "Inversion":            lambda r: apply_inversion(r, params.get('inversion_mix', 1.0)),
            "Postérisation Warhol": lambda r: poster_solarize(r, params.get('poster_levels', 4), params.get('poster_mix', 1.0)),
            "Cross-Process":        lambda r: cross_process(r, params.get('cross_mix', 0.9)),
            "Duotone":              lambda r: duotone(r, params.get('duo_shadows', (0.05, 0.0, 0.2)), params.get('duo_highlights', (1.0, 0.9, 0.1)), params.get('duo_mix', 1.0)),
            "Distorsion liquide":   lambda r: apply_distortion(r, params.get('dist_intensity', 0.4), params.get('dist_freq', 6), params.get('dist_mix', 0.8)),
            "Aberration Chromatique": lambda r: chromatic_aberration(r, params.get('ca_strength', 10), params.get('ca_falloff', 0.6), params.get('ca_mix', 1.0)),
            "Diffraction Spectrale": lambda r: spectral_diffraction(r, params.get('diff_freq', 30), params.get('diff_disp', 0.12), mix=params.get('diff_mix', 0.8)),
            "Holographique":        lambda r: holographic_effect(r, params.get('holo_irid', 0.5), params.get('holo_par', 0.1)),
            "Néon":                 lambda r: neon_effect(r, params.get('neon_hue', 0.75), params.get('neon_int', 2.0), params.get('neon_mix', 0.85), params.get('neon_sat', 0.95), params.get('neon_spread', 2.5), params.get('neon_thin', 0.6), params.get('neon_dark', 0.35)),
            "Solarisation":         lambda r: solarize(r, params.get('sol_threshold', 0.5), params.get('sol_invert_above', True), params.get('sol_channels', None), params.get('sol_mix', 1.0)),
            "Trame Halftone":       lambda r: halftone(r, params.get('half_size', 7), params.get('half_angle', 45), params.get('half_style', 'dots'), params.get('half_mix', 1.0)),
            "Glitch RGB":           lambda r: glitch_rgb(r, params.get('glitch_h', 12), params.get('glitch_v', 4), params.get('glitch_bands', 8), params.get('glitch_int', 1.0)),
            "Vignette + Grain":     lambda r: vignette_grain(r, params.get('vig_str', 0.6), params.get('grain_amt', 0.06), params.get('grain_sz', 1), params.get('vg_mix', 1.0)),
        }

        for idx, effect in enumerate(effects):
            progress.progress((idx + 1) / len(effects))
            if effect in DISPATCH:
                result = DISPATCH[effect](result)
            else:
                st.warning(f"Effet inconnu ignoré : {effect}")

        final = np.clip(img_array * (1 - params['global_mix']) + result * params['global_mix'], 0, 1)
        progress.empty()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_input}_{timestamp}.{download_format.lower()}"
        final_u8 = (final * 255).astype(np.uint8)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(final_u8, use_container_width=True, caption="RENDU FINAL")
        with col2:
            st.download_button(
                "📥 Exporter",
                image_to_bytes(final_u8, download_format, QUALITY),
                file_name=filename,
                mime=f"image/{download_format.lower()}"
            )
            st.markdown("**RGB moyen :**")
            st.write(f"R {final_u8[...,0].mean():.0f}  G {final_u8[...,1].mean():.0f}  B {final_u8[...,2].mean():.0f}")

    except Exception as e:
        st.error(f"Erreur de traitement : {e}")
        st.stop()

elif uploaded_file and not effects:
    st.info("Sélectionnez au moins un effet dans le panneau gauche.")
else:
    st.info("⬅️ Chargez une image et sélectionnez des effets pour commencer.")

st.markdown("---")
st.markdown("**VMC Poster FX** — v2.1")
