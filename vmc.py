import numpy as np
import streamlit as st
from scipy import ndimage
from PIL import Image
import io

# Configuration de la page
st.set_page_config(page_title="Détection de contours", layout="wide")
st.title("Détection de contours avancée")

# Fonctions utilitaires
def image_to_bytes(img_array, format='JPEG'):
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

# Sidebar controls
with st.sidebar:
    st.header("Paramètres")
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
    filter_type = st.multiselect(
        "Filtres à appliquer",
        ['Sobel', 'Prewitt', 'Roberts'],
        default=['Sobel', 'Prewitt', 'Roberts']
    )
    auto_threshold = st.checkbox("Seuil automatique (Otsu)", value=True)
    apply_gaussian = st.checkbox("Pré-filtre Gaussien")
    gaussian_sigma = st.slider("Intensité Gaussien", 0.0, 3.0, 1.0, disabled=not apply_gaussian)
    download_format = st.radio("Format de téléchargement", ['JPEG', 'PNG'])

if uploaded_file is not None:
    # Traitement de l'image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if img_array.ndim == 3:
        img_gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        img_gray = img_array
    
    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())
    
    # Application du filtre Gaussien
    if apply_gaussian:
        img_gray = ndimage.gaussian_filter(img_gray, sigma=gaussian_sigma)

    # Définition des filtres
    kernels = {
        'Sobel': {
            'h': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4,
            'v': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4
        },
        'Prewitt': {
            'h': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 3,
            'v': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) / 3
        },
        'Roberts': {
            'h': np.array([[1, 0], [0, -1]]),
            'v': np.array([[0, 1], [-1, 0]])
        }
    }

    # Calcul des gradients seulement pour les filtres sélectionnés
    results = {}
    thresholds = {}
    
    # Interface des seuils manuels
    if not auto_threshold:
        with st.sidebar:
            st.subheader("Seuils manuels")
            manual_thresholds = {}
            for name in filter_type:
                manual_thresholds[name] = st.slider(
                    f"Seuil {name}",
                    0.0, 1.0, 0.5,
                    key=f"th_{name}"
                )

    # Calcul des résultats
    for name in filter_type:
        h = ndimage.convolve(img_gray, kernels[name]['h'], mode='nearest')
        v = ndimage.convolve(img_gray, kernels[name]['v'], mode='nearest')
        results[name] = np.sqrt(h**2 + v**2)
        
        # Calcul du seuil
        if auto_threshold:
            # Algorithme d'Otsu optimisé
            hist, bins = np.histogram(results[name] * 255, bins=256)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            hist = hist.astype(float)
            
            total = hist.sum()
            if total == 0:
                thresholds[name] = 0.5
                continue
                
            norm_hist = hist / total
            mean = (norm_hist * bin_centers).sum()
            max_var = best_thresh = 0
            
            cumulative_sum = np.cumsum(norm_hist)
            cumulative_mean = np.cumsum(norm_hist * bin_centers)
            
            for i in range(1, 256):
                w0 = cumulative_sum[i]
                w1 = 1 - w0
                if w0 == 0 or w1 == 0:
                    continue
                
                mean0 = cumulative_mean[i] / w0
                mean1 = (mean - cumulative_mean[i]) / w1
                var = w0 * w1 * (mean0 - mean1)**2
                
                if var > max_var:
                    max_var = var
                    best_thresh = i
            
            thresholds[name] = best_thresh / 255
        else:
            thresholds[name] = manual_thresholds[name]

    # Application des seuils
    binary_results = {name: (results[name] > thresholds[name]) for name in results}

    # Affichage des résultats
    st.header("Résultats de détection")
    cols = st.columns(len(filter_type))
    
    for col, name in zip(cols, filter_type):
        with col:
            # Affichage de l'image
            st.image(
                binary_results[name], 
                use_column_width=True, 
                caption=f"{name} (Seuil: {thresholds[name]:.3f})"
            )
            
            # Bouton de téléchargement
            img_bytes = image_to_bytes(binary_results[name], format=download_format)
            st.download_button(
                label=f"Télécharger {name}",
                data=img_bytes,
                file_name=f"{name}_edges.{download_format.lower()}",
                mime=f"image/{download_format.lower()}"
            )

    # Affichage des paramètres utilisés
    with st.expander("Afficher les paramètres utilisés"):
        st.write(f"Filtres appliqués: {', '.join(filter_type)}")
        st.write(f"Méthode de seuil: {'Automatique (Otsu)' if auto_threshold else 'Manuel'}")
        if apply_gaussian:
            st.write(f"Filtre Gaussien (σ={gaussian_sigma})")
        st.write(f"Format d'export: {download_format}")

else:
    st.info("Veuillez télécharger une image pour commencer l'analyse")
