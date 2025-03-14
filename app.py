import streamlit as st
import face_recognition
import cv2
import os
import numpy as np
from PIL import Image
import io
import base64

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="Riconosciamo Giorgia Meloni",
    page_icon="👩‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS personalizzato
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #3366cc;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px #cccccc;
    }
    .subtitle {
        font-size: 20px;
        color: #666666;
        text-align: center;
        margin-bottom: 30px;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin-bottom: 20px;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #e2f0fd;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin-bottom: 20px;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .download-btn {
        background-color: #3366cc !important;
    }
    .stSlider {
        padding-top: 15px;
        padding-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

def get_image_download_link(img, filename="risultato_meloni.jpg", text="Scarica l'immagine"):
    """Genera un link per scaricare l'immagine"""
    buffered = io.BytesIO()
    img_pil = Image.fromarray(img)
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" style="display: inline-block; padding: 10px 20px; background-color: #3366cc; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; text-align: center; width: 100%;">{text}</a>'
    return href

def create_meloni_detector(reference_images_paths):
    """Crea un rilevatore per Giorgia Meloni basato su immagini di riferimento locali."""
    known_meloni_encodings = []
    
    with st.status("⏳ Caricamento delle immagini di riferimento di Giorgia Meloni...", expanded=False) as status:
        # Processa ogni immagine di riferimento
        for i, image_path in enumerate(reference_images_paths):
            try:
                status.update(label=f"⏳ Elaborazione immagine {i+1}/{len(reference_images_paths)}")
                
                # Carica l'immagine
                image = cv2.imread(image_path)
                if image is None:
                    st.error(f"Impossibile caricare l'immagine {image_path}. Verifica che il percorso sia corretto.")
                    continue
                
                # Face recognition funziona con RGB, ma OpenCV usa BGR
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Trova i volti nell'immagine
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
                
                if not face_locations:
                    status.update(label=f"⚠️ Nessun volto trovato nell'immagine {i+1}")
                    continue
                
                # Prendi il primo volto trovato (assumiamo sia Meloni)
                encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if encodings:
                    known_meloni_encodings.append(encodings[0])
                    status.update(label=f"✅ Encoding estratto con successo dall'immagine {i+1}")
                else:
                    status.update(label=f"❌ Impossibile estrarre encoding dall'immagine {i+1}")
            
            except Exception as e:
                status.update(label=f"❌ Errore nell'immagine {i+1}: {e}")
        
        if not known_meloni_encodings:
            status.update(label="❌ Impossibile creare rilevatore: nessun encoding valido", state="error")
            return None
        
        status.update(label=f"✅ Rilevatore creato con {len(known_meloni_encodings)} encodings", state="complete")
    
    return known_meloni_encodings

def detect_meloni(image, known_meloni_encodings, tolerance=0.6):
    """
    Rileva Giorgia Meloni in un'immagine.
    
    Args:
        image: Può essere un percorso file o un'immagine già caricata (numpy array)
        known_meloni_encodings: Lista di encodings facciali di riferimento
        tolerance: Soglia di corrispondenza (più basso = più preciso)
        
    Returns:
        Tuple (immagine con box, booleano se Meloni è stata trovata, numero di volti trovati)
    """
    meloni_trovata = False
    num_volti = 0
    
    # Controlla se l'input è un percorso o un'immagine già caricata
    if isinstance(image, str):
        try:
            # E' un percorso file
            img = cv2.imread(image)
            if img is None:
                st.error(f"Impossibile caricare l'immagine {image}. Verifica che il percorso sia corretto.")
                return None, False, 0
        except Exception as e:
            st.error(f"Errore durante il caricamento dell'immagine: {e}")
            return None, False, 0
    else:
        # E' già un'immagine caricata
        img = image.copy()
    
    # Converti in RGB (face_recognition usa RGB)
    if len(img.shape) == 3 and img.shape[2] == 3:  # Verifica che sia un'immagine a colori
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = img  # L'immagine potrebbe essere già in RGB
    
    # Trova tutti i volti nell'immagine
    with st.spinner("🔍 Ricerca volti nell'immagine..."):
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        num_volti = len(face_locations)
    
    if not face_locations:
        return img, False, 0
    
    # Calcola gli encodings per ogni volto trovato
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # Confronta ogni volto con gli encodings di Meloni
    for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
        # Controlla se il volto corrisponde a Meloni
        matches = face_recognition.compare_faces(known_meloni_encodings, face_encoding, tolerance=tolerance)
        
        # Se almeno un riferimento corrisponde, è probabilmente Meloni
        if True in matches:
            meloni_trovata = True
            # Estrai le coordinate
            top, right, bottom, left = face_location
            
            # Disegna un rettangolo
            color = (0, 0, 255)  # Rosso in BGR
            thickness = 3
            cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
            
            # Aggiungi etichetta con sfondo
            label = "Giorgia Meloni"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            font_thickness = 2
            
            # Calcola le dimensioni del testo
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Disegna sfondo per il testo
            cv2.rectangle(img, (left, top - text_height - 10), (left + text_width, top), (0, 0, 255), -1)
            
            # Aggiungi testo
            cv2.putText(img, label, (left, top - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return img, meloni_trovata, num_volti

def process_uploaded_image(uploaded_file):
    """Elabora un file caricato in formato immagine numpy array."""
    # Leggi il file caricato
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decodifica l'immagine
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

def main():
    # Titolo e sottotitolo con stile
    st.markdown('<div class="main-title">👩‍💼 Riconoscitore Facciale di Giorgia Meloni</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Questa app individua il volto di Giorgia Meloni nelle foto e lo evidenzia con un box rosso.</div>', unsafe_allow_html=True)
    
    # Barra laterale con impostazioni
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Giorgia_Meloni_Official_Portrait.jpg", 
                 caption="Giorgia Meloni", use_container_width=True)
        
        st.markdown("## ⚙️ Impostazioni")
        
        # Slider per la precisione
        tolerance = st.slider(
            "Precisione del riconoscimento", 
            min_value=0.4, 
            max_value=0.8, 
            value=0.6, 
            step=0.05,
            help="Valori più bassi = riconoscimento più preciso, ma potrebbero esserci più falsi negativi"
        )
        
        # Opzioni per l'utente
        option = st.radio("Scegli un'opzione:", ["Usa immagine di default", "Carica un'immagine"])
        
        # Informazioni
        st.markdown("---")
        st.markdown("### 📝 Come funziona")
        st.markdown("""
        1. L'app utilizza il modello di riconoscimento facciale **face_recognition**
        2. Confronta i volti rilevati con quelli di riferimento di Giorgia Meloni
        3. Identifica e marca con un box rosso i volti che corrispondono
        """)
    
    # Definisci i percorsi delle immagini di riferimento di Giorgia Meloni
    # Assicurati che i percorsi siano corretti (uso / invece di \ per compatibilità)
    reference_images = [
        "reference_images/meloni1.jpg",
        "reference_images/meloni2.jpg",
        "reference_images/meloni3.jpg",
        "reference_images/meloni4.jpg",
        "reference_images/meloni5.jpg",
        "reference_images/meloni6.jpg",
        "reference_images/meloni.jpg",
    ]
    
    # Filtra solo i percorsi che esistono
    existing_references = [path for path in reference_images if os.path.exists(path)]
    if not existing_references:
        st.markdown('<div class="error-box">❌ Nessuna immagine di riferimento trovata! Verifica i percorsi delle immagini.</div>', unsafe_allow_html=True)
        st.stop()
    
    # Inizializza il rilevatore
    try:
        detector_placeholder = st.empty()
        with detector_placeholder.container():
            known_meloni_encodings = create_meloni_detector(existing_references)
            if not known_meloni_encodings:
                st.markdown('<div class="error-box">❌ Impossibile creare il rilevatore. Verifica le immagini di riferimento.</div>', unsafe_allow_html=True)
                st.stop()
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Errore nella creazione del rilevatore: {e}</div>', unsafe_allow_html=True)
        st.stop()
    
    # Nascondi il placeholder dopo l'inizializzazione
    detector_placeholder.empty()
    
    # Colonne per l'interfaccia
    col1, col2 = st.columns(2)
    
    # Gestisci le diverse opzioni
    if option == "Usa immagine di default":
        # Percorso dell'immagine di test (uso / invece di \ per compatibilità)
        test_image_path = "test_image/meloni_in_gruppo.jpg"
        
        if not os.path.exists(test_image_path):
            col1.markdown(f'<div class="error-box">❌ Immagine di default non trovata: {test_image_path}</div>', unsafe_allow_html=True)
        else:
            # Container per l'immagine originale
            with col1.container():
                st.markdown("### 📷 Immagine originale")
                # Mostra l'immagine originale
                image = cv2.imread(test_image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(rgb_image, caption="", use_container_width=True)
                
                # Pulsante per l'analisi
                analyze_button = st.button("🔍 Analizza immagine", key="analyze_default")
            
            # Container per il risultato
            with col2.container():
                st.markdown("### 🔎 Risultato dell'analisi")
                
                result_placeholder = st.empty()
                
                if analyze_button:
                    with st.spinner("🔄 Analisi in corso..."):
                        # Rileva Meloni nell'immagine
                        result_image, meloni_trovata, num_volti = detect_meloni(test_image_path, known_meloni_encodings, tolerance)
                        
                        if result_image is not None:
                            # Converti in RGB per visualizzare
                            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                            
                            # Mostra l'immagine risultante
                            result_placeholder.image(result_rgb, use_container_width=True)
                            
                            # Mostra il risultato testuale
                            if meloni_trovata:
                                st.markdown(f'<div class="success-box">✅ <b>Giorgia Meloni è stata identificata!</b><br>Rilevati un totale di {num_volti} volti nell\'immagine.</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="error-box">⚠️ <b>Giorgia Meloni non è stata trovata</b><br>Rilevati un totale di {num_volti} volti nell\'immagine.</div>', unsafe_allow_html=True)
                            
                            # Pulsante per scaricare il risultato
                            st.markdown(get_image_download_link(result_rgb, "giorgia_meloni_rilevata.jpg", "📥 Scarica l'immagine risultante"), unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-box">❌ Elaborazione dell\'immagine fallita.</div>', unsafe_allow_html=True)
    
    else:  # Carica un'immagine
        # Container per l'upload
        with col1.container():
            st.markdown("### 📤 Carica la tua immagine")
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Prepara l'immagine
                image = process_uploaded_image(uploaded_file)
                
                # Mostra l'immagine caricata
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(rgb_image, use_container_width=True)
                
                # Pulsante per l'analisi
                analyze_button = st.button("🔍 Analizza immagine", key="analyze_upload")
            else:
                st.markdown('<div class="info-box">ℹ️ Carica un\'immagine JPG o PNG per iniziare l\'analisi.</div>', unsafe_allow_html=True)
                analyze_button = False
        
        # Container per il risultato
        with col2.container():
            st.markdown("### 🔎 Risultato dell'analisi")
            
            result_placeholder = st.empty()
            
            if uploaded_file is not None and analyze_button:
                with st.spinner("🔄 Analisi in corso..."):
                    # Rileva Meloni nell'immagine
                    result_image, meloni_trovata, num_volti = detect_meloni(image, known_meloni_encodings, tolerance)
                    
                    if result_image is not None:
                        # Converti in RGB per visualizzare
                        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        
                        # Mostra l'immagine risultante
                        result_placeholder.image(result_rgb, use_container_width=True)
                        
                        # Mostra il risultato testuale
                        if meloni_trovata:
                            st.markdown(f'<div class="success-box">✅ <b>Giorgia Meloni è stata identificata!</b><br>Rilevati un totale di {num_volti} volti nell\'immagine.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="error-box">⚠️ <b>Giorgia Meloni non è stata trovata</b><br>Rilevati un totale di {num_volti} volti nell\'immagine.</div>', unsafe_allow_html=True)
                        
                        # Pulsante per scaricare il risultato
                        st.markdown(get_image_download_link(result_rgb, "giorgia_meloni_rilevata.jpg", "📥 Scarica l'immagine risultante"), unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-box">❌ Elaborazione dell\'immagine fallita.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()