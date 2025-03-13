import numpy as np
import streamlit as st
from scipy import ndimage
from PIL import Image
import io

# Configuration de la page
st.set_page_config(page_title="Détection de contours", layout="wide")
st.title("Détection de contours avancée")

def image_to_bytes(img_array, format='JPEG'):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

with st.sidebar:
    st.header("Paramètres")
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
    
    # Garantir au moins 1 filtre sélectionné
    filter_type = st.multiselect(
        "Filtres à appliquer",
        ['Sobel', 'Prewitt', 'Roberts'],
        default=['Sobel']
    )
    
    # Validation de la sélection
    if not filter_type:
        st.error("Veuillez sélectionner au moins un filtre")
        st.stop()
    
    auto_threshold = st.checkbox("Seuil automatique (Otsu)", value=True)
    apply_gaussian = st.checkbox("Pré-filtre Gaussien")
    gaussian_sigma = st.slider("Intensité Gaussien", 0.0, 3.0, 1.0, disabled=not apply_gaussian)
    download_format = st.radio("Format de téléchargement", ['JPEG', 'PNG'])

if uploaded_file is not None and filter_type:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if img_array.ndim == 3:
        img_gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        img_gray = img_array
    
    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())
    
    if apply_gaussian:
        img_gray = ndimage.gaussian_filter(img_gray, sigma=gaussian_sigma)

    kernels = {
        'Sobel': {'h': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/4, 
                  'v': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])/4},
        'Prewitt': {'h': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])/3,
                    'v': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])/3},
        'Roberts': {'h': np.array([[1, 0], [0, -1]]),
                    'v': np.array([[0, 1], [-1, 0]])}
    }

    # Calcul des résultats
    results = {}
    thresholds = {}
    
    if not auto_threshold:
        with st.sidebar:
            st.subheader("Seuils manuels")
            manual_thresholds = {name: st.slider(f"Seuil {name}", 0.0, 1.0, 0.5) for name in filter_type}

    for name in filter_type:
        h = ndimage.convolve(img_gray, kernels[name]['h'], mode='nearest')
        v = ndimage.convolve(img_gray, kernels[name]['v'], mode='nearest')
        results[name] = np.sqrt(h**2 + v**2)
        
        if auto_threshold:
            hist = np.histogram(results[name], bins=256, range=(0, 1))[0]
            total = hist.sum()
            if total == 0:
                thresholds[name] = 0.5
                continue
                
            sum_total = np.arange(256) @ hist
            max_var = best_thresh = 0
            sum_back = 0
            weight_back = 0
            
            for t in range(256):
                weight_back += hist[t]
                if weight_back == 0:
                    continue
                weight_front = total - weight_back
                if weight_front == 0:
                    break
                sum_back += t * hist[t]
                mean_back = sum_back / weight_back
                mean_front = (sum_total - sum_back) / weight_front
                var = weight_back * weight_front * (mean_back - mean_front)**2
                if var > max_var:
                    max_var = var
                    best_thresh = t
            thresholds[name] = best_thresh / 255
        else:
            thresholds[name] = manual_thresholds[name]

    binary_results = {name: (results[name] > thresholds[name]) for name in filter_type}

    # Affichage corrigé
    st.header("Résultats")
    cols = st.columns(len(filter_type))
    
    for col, name in zip(cols, filter_type):
        with col:
            st.image(
                binary_results[name], 
                use_column_width=True,
                caption=f"{name} (Seuil: {thresholds[name]:.3f})"
            )
            img_bytes = image_to_bytes(binary_results[name], download_format)
            st.download_button(
                f"Télécharger {name}",
                img_bytes,
                file_name=f"edges_{name.lower()}.{download_format.lower()}",
                mime=f"image/{download_format.lower()}"
            )

else:
    if uploaded_file and not filter_type:
        st.error("Configuration invalide : aucun filtre sélectionné")
    else:
        st.info("Veuillez télécharger une image et sélectionner au moins un filtre")
