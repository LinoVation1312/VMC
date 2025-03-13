import numpy as np
import streamlit as st
from scipy import ndimage
from PIL import Image
import io
import matplotlib.pyplot as plt

# Configuration de la page avec style VMC
st.set_page_config(
    page_title="VMC Edge Detector",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown(f"""
    <style>
    .main {{
        background-color: #0E1117;
        color: #FAFAFA;
    }}
    .st-emotion-cache-6qob1r {{
        background-color: #1A1D24 !important;
    }}
    h1 {{
        color: #FF4B4B !important;
        font-family: 'Helvetica Neue', sans-serif;
        text-shadow: 0 0 10px #FF4B4B44;
    }}
    .stDownloadButton button {{
        background: #FF4B4B !important;
        border: 1px solid #FF4B4B !important;
        color: black !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
    }}
    .stDownloadButton button:hover {{
        background: #FF3333 !important;
        transform: scale(1.05);
        box-shadow: 0 0 15px #FF4B4B;
    }}
    .element-container img {{
        border-radius: 5px;
        border: 2px solid #FF4B4B !important;
        box-shadow: 0 0 20px #FF4B4B33;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Header VMC
st.image("https://i.ibb.co/0jq6Y3N/vmc-logo.png", use_container_width=300) 
st.title("VMC Visual Processor")
st.markdown("**Outils de traitement visuel pour performances live** 🎧⚡")

def image_to_bytes(img_array, format='JPEG'):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

with st.sidebar:
    st.header("🎛️ Contrôles VMC")
    
    # Section Upload
    with st.container(border=True):
        st.subheader("🖼️ Source Audio-Visuelle")
        uploaded_file = st.file_uploader("Charger un sample visuel", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    # Section Filtres
    with st.container(border=True):
        st.subheader("🔊 Paramètres de Filtration")
        filter_type = st.multiselect(
            "Filtres actifs",
            ['Sobel', 'Prewitt', 'Roberts'],
            default=['Sobel', 'Roberts'],
            format_func=lambda x: f'⚡ {x}'
        )
        
        auto_threshold = st.checkbox("Auto-bpm (Otsu)", value=True)
        
        if not auto_threshold:
            st.subheader("🎚️ Niveaux manuels")
            manual_thresholds = {name: st.slider(f"{name}", 0.0, 1.0, 0.5) for name in filter_type}
        
        st.subheader("🌀 Effets")
        apply_gaussian = st.checkbox("Reverb Gaussien")
        if apply_gaussian:
            gaussian_sigma = st.slider("Intensité", 0.0, 3.0, 1.0)

# Processing section
if uploaded_file and filter_type:
    with st.spinner("Processing audio-visual stream..."):
        # Chargement et conversion
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        # Conversion en niveaux de gris avec normalisation robuste
        if img_array.ndim == 3:
            img_gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(float)
        else:
            img_gray = img_array.astype(float)
            
        img_gray = (img_gray - np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray) + 1e-8)  # Évite la division par zéro
        
        # Pré-filtrage Gaussien
        if apply_gaussian:
            img_gray = ndimage.gaussian_filter(img_gray, sigma=gaussian_sigma)
        
        # Nouveaux kernels non normalisés
        kernels = {
            'Sobel': {
                'h': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                'v': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            },
            'Prewitt': {
                'h': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                'v': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            },
            'Roberts': {
                'h': np.array([[1, 0], [0, -1]]),
                'v': np.array([[0, 1], [-1, 0]])
            }
        }

        results = {}
        thresholds = {}
        
        for name in filter_type:
            # Calcul des gradients
            h = ndimage.convolve(img_gray, kernels[name]['h'], mode='nearest')
            v = ndimage.convolve(img_gray, kernels[name]['v'], mode='nearest')
            
            # Calcul de la magnitude avec normalisation adaptative
            grad_mag = np.sqrt(h**2 + v**2)
            grad_mag = (grad_mag - np.min(grad_mag)) / (np.max(grad_mag) - np.min(grad_mag) + 1e-8)
            results[name] = grad_mag * 1.5  # Boost techno contrôlé
            
            # Calcul du seuil Otsu amélioré
            if auto_threshold:
                hist, bins = np.histogram(results[name] * 255, bins=256, range=(0, 255))
                bin_centers = (bins[:-1] + bins[1:]) / 2
                total = hist.sum()
                
                if total == 0:
                    thresholds[name] = 0.5
                    continue
                
                norm_hist = hist / total
                cumulative = np.cumsum(norm_hist)
                cumulative_mean = np.cumsum(norm_hist * bin_centers)
                
                max_var = best_thresh = 0
                for t in range(1, 255):
                    w0, w1 = cumulative[t], 1 - cumulative[t]
                    if w0 < 1e-8 or w1 < 1e-8:
                        continue
                    mean0 = cumulative_mean[t] / w0
                    mean1 = (cumulative_mean[-1] - cumulative_mean[t]) / w1
                    var = w0 * w1 * (mean0 - mean1)**2
                    if var > max_var:
                        max_var, best_thresh = var, t
                thresholds[name] = best_thresh / 255
            else:
                thresholds[name] = manual_thresholds[name]

        # Application des seuils avec vérification
        binary_results = {}
        for name in filter_type:
            threshold = np.clip(thresholds[name], 0.01, 0.99)  # Évite les seuils extrêmes
            binary_results[name] = (results[name] > threshold).astype(float)

    # Affichage des résultats
    st.header("📡 Sortie Visuelle Live", divider="red")
    cols = st.columns(len(filter_type))
    
    for col, name in zip(cols, filter_type):
        with col:
            with st.container(border=True):
                # Vignette stylisée
                st.markdown(f"#### {name} `v{np.random.uniform(1.0, 3.0):.1f}`")
                st.image(
                    binary_results[name], 
                    use_container_width=True,
                    caption=f"Seuil: {thresholds[name]:.3f} | BPM: {np.random.randint(120, 150)}"
                )
                
                # Bouton de téléchargement
                img_bytes = image_to_bytes(binary_results[name], 'PNG')
                st.download_button(
                    f"Exporter {name}",
                    img_bytes,
                    file_name=f"vmc_{name.lower()}_{np.random.randint(1000,9999)}.png",
                    mime="image/png",
                    use_container_width=True
                )

    # Visualisation des gradients
    with st.expander("🔬 Analyse de Fréquence"):
        fig, ax = plt.subplots(figsize=(10, 4))
        for name in filter_type:
            ax.plot(results[name].mean(axis=0), label=name)
        ax.set_title("Profil de Fréquence Horizontal")
        ax.set_facecolor('#0E1117')
        ax.grid(color='#FF4B4B33')
        plt.legend()
        st.pyplot(fig)

elif uploaded_file and not filter_type:
    st.error("⚠️ Configuration audio-visuelle incomplète : sélectionnez au moins un filtre!")
else:
    st.info("🎧 Connectez un sample visuel pour initier le traitement...", icon="⚠️")

# Footer VMC
st.markdown("---")
st.markdown("""
    **VMC Visual Tools**  
    *Outils pour performances audiovisuelles live*  
    [GitHub](https://github.com/vmc) | [SoundCloud](https://soundcloud.com/vmc) | [Bandcamp](https://vmc.bandcamp.com)
""")
