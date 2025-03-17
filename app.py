import streamlit as st
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import os
import time
import io
import base64

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="🇮🇹 Meloni Detector",
    page_icon="🇮🇹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stile CSS personalizzato con miglioramento della leggibilità
st.markdown("""
<style>
    /* Stili generali */
    body {
        color: #ffffff;
        background-color: #0e1117;
    }
    
    .main-title {
        color: #008C45; /* verde bandiera italiana */
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-title {
        color: #CD212A; /* rosso bandiera italiana */
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Box per messaggi */
    .success-box {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        font-weight: bold;
        font-size: 16px;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        font-weight: bold;
        font-size: 16px;
    }
    
    .info-box {
        background-color: #e6f7ff;
        border-left: 6px solid #1890ff;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        color: #0c5460;
        font-weight: bold;
        font-size: 16px;
    }
    
    /* Miglioramento leggibilità dei panel bianchi */
    .panel {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
        color: #333333;
        font-size: 16px;
        font-weight: 500;
    }
    
    /* Stile del pulsante di analisi */
    .stButton button {
        background-color: #1E88E5 !important; /* blu più gradevole */
        color: white !important;
        font-weight: bold !important;
        border-radius: 20px !important;
        padding: 0.5rem 2rem !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
        font-size: 18px !important;
    }
    
    .stButton button:hover {
        background-color: #1565C0 !important; /* blu più scuro all'hover */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Stile per il pulsante di download */
    .download-button {
        margin-top: 1rem;
    }
    
    /* Stile per il separatore */
    hr {
        margin: 2rem 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.2), rgba(0,0,0,0));
    }
    
    /* Stile per i titoli delle sezioni */
    .section-title {
        color: #f8f9fa;
        border-bottom: 2px solid #008C45;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    /* Badge italiano */
    .italian-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 50px;
        background: linear-gradient(90deg, #008C45 33%, #F4F9F0 33%, #F4F9F0 66%, #CD212A 66%);
        color: white;
        font-weight: bold;
        margin-right: 10px;
    }
    
    /* Migliorare leggibilità messaggi informativi */
    .stAlert > div {
        font-size: 16px !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
    }
    
    /* Evidenziare le metriche */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 18px !important;
        font-weight: bold !important;
        color: #f8f9fa !important;
    }
    
    .metric-value {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #4CAF50 !important;
    }
    
    /* Migliorare visibilità della legenda */
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 5px;
        color: #333;
        font-weight: 500;
    }
    
    .legend-color {
        width: 20px;
        height: 20px;
        margin-right: 10px;
        border: 1px solid rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Funzione per il riconoscimento facciale
def recognize_meloni(image, known_encodings, tolerance=0.58):
    # Cerca tutti i volti nell'immagine
    face_locations = face_recognition.face_locations(image)
    
    # Se non ci sono volti, restituisci l'immagine originale
    if not face_locations:
        return Image.fromarray(image), [], "Nessun volto trovato nell'immagine."
    
    # Ottieni le codifiche per tutti i volti trovati
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Crea un'immagine PIL per il disegno
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Lista per tenere traccia dei volti di Meloni trovati
    meloni_faces = []
    
    # Controlla ogni volto
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Confronta con le codifiche note di Meloni
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        
        # Calcola la confidenza
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        confidence = 1 - face_distances[best_match_index]
        
        # Se è una corrispondenza
        if True in matches:
            # Disegna un rettangolo verde
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=4)
            
            # Aggiungi etichetta
            label = f"Meloni ({confidence:.1%})"
            text_height = 20
            
            # Rettangolo per il testo
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), 
                          fill=(0, 255, 0), outline=(0, 255, 0))
            
            # Testo
            draw.text((left + 6, bottom - text_height - 5), label, fill=(0, 0, 0))
            
            # Aggiungi alla lista di volti di Meloni
            meloni_faces.append({
                "location": (top, right, bottom, left),
                "confidence": confidence
            })
        else:
            # Disegna un rettangolo rosso per i non-Meloni
            draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)
    
    # Prepara il messaggio di risultato
    if meloni_faces:
        result_message = f"Trovate {len(meloni_faces)} istanze di Meloni nell'immagine!"
    else:
        result_message = "Meloni non trovata nell'immagine."
    
    return pil_image, meloni_faces, result_message

# Funzione per caricare le codifiche facciali di Meloni
@st.cache_data
def load_meloni_encodings(folder_path="meloni_images"):
    encodings = []
    image_count = 0
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        st.warning(f"Cartella {folder_path} creata. Inserisci almeno un'immagine di Meloni.")
        return encodings, image_count
    
    # Carica tutte le immagini nella cartella
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image_count += 1
            
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    encodings.append(face_encodings[0])
            except Exception as e:
                st.error(f"Errore nel caricamento di {filename}: {str(e)}")
                
    return encodings, image_count

# Main
def main():
    # Intestazione
    st.markdown('<p class="main-title">🇮🇹 Meloni Detector 🇮🇹</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Riconosci Giorgia Meloni in qualsiasi immagine</p>', unsafe_allow_html=True)
    
    # Banner tricolore
    st.markdown("""
    <div style="background: linear-gradient(90deg, #008C45 33%, #F4F9F0 33%, #F4F9F0 66%, #CD212A 66%); 
                height: 10px; margin-bottom: 2rem; border-radius: 5px;">
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Flag_of_Italy.svg/320px-Flag_of_Italy.svg.png", use_container_width=True)
        
        st.markdown('<p class="section-title">Impostazioni</p>', unsafe_allow_html=True)
        
        # Carica le codifiche di Meloni
        meloni_encodings, image_count = load_meloni_encodings()
        
        if len(meloni_encodings) == 0:
            st.warning("⚠️ Nessuna immagine valida trovata nella cartella 'meloni_images'.")
            st.info("Inserisci alcune immagini di Giorgia Meloni nella cartella 'meloni_images' per iniziare.")
            st.stop()
        else:
            st.success(f"✅ Caricate {len(meloni_encodings)} codifiche facciali da {image_count} immagini.")
        
        # Impostazioni di tolleranza con tooltip più descrittivo
        tolerance = st.slider(
            "Livello di sensibilità",
            min_value=0.1,
            max_value=1.0,
            value=0.58,
            step=0.01,
            help="Regola la sensibilità del riconoscimento. Valori più bassi = riconoscimento più preciso ma meno sensibile. Valori più alti = più possibilità di identificare Meloni, ma anche più falsi positivi."
        )
        
        # Visualizzazione dello stato di sensibilità
        if tolerance < 0.4:
            st.info("🔍 Sensibilità: Molto precisa")
        elif tolerance < 0.55:
            st.info("🔍 Sensibilità: Precisa")
        elif tolerance < 0.7:
            st.info("🔍 Sensibilità: Bilanciata")
        else:
            st.info("🔍 Sensibilità: Alta (possibili falsi positivi)")
        
        # Modalità demo
        st.markdown("---")
        use_demo = st.checkbox("Usa immagine di dimostrazione", value=True)
        
        # Crediti
        st.markdown("---")
        st.markdown("### Info")
        st.info(
            "Questo tool utilizza face_recognition per identificare Giorgia Meloni nelle immagini. "
            "Le immagini vengono elaborate localmente e non vengono caricate su server esterni."
        )
    
    # Layout principale
    col1, col2 = st.columns([1, 1], gap="large")
    
    # Colonna di input
    with col1:
        st.markdown('<p class="section-title">1. Seleziona un\'immagine</p>', unsafe_allow_html=True)
        
        # Descrizione migliorata e più visibile
        st.markdown("""
        <div class="panel" style="background-color: #e6f7ff; border-left: 6px solid #1890ff;">
        <strong>Istruzioni:</strong> Puoi usare l'immagine demo o caricare la tua immagine personale per cercare Giorgia Meloni. 
        Le foto con volti chiari e ben illuminati daranno i migliori risultati.
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_image = None
        
        # Usa l'immagine demo o permetti upload
        if use_demo:
            if os.path.exists("foto_gruppo.jpg"):
                uploaded_image = "foto_gruppo.jpg"
                st.markdown("""
                <div class="panel" style="color: #155724; background-color: #d4edda; border-left: 6px solid #28a745;">
                <strong>✅ Usando l'immagine di dimostrazione</strong><br>
                L'immagine di gruppo predefinita è stata caricata con successo.
                </div>
                """, unsafe_allow_html=True)
                
                # Contenitore per l'immagine
                st.image("foto_gruppo.jpg", use_container_width=True, caption="Immagine di dimostrazione")
            else:
                st.markdown("""
                <div class="panel" style="color: #721c24; background-color: #f8d7da; border-left: 6px solid #dc3545;">
                <strong>⚠️ Immagine demo 'foto_gruppo.jpg' non trovata!</strong><br>
                Inserisci un'immagine chiamata 'foto_gruppo.jpg' nella cartella del progetto o carica un'immagine personalizzata.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Carica una foto", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                uploaded_image = Image.open(io.BytesIO(image_bytes))
                st.image(uploaded_image, caption="Immagine caricata", use_container_width=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 20px; color: #0c5460; background-color: #d1ecf1; border-radius: 5px; margin-top: 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" viewBox="0 0 16 16" style="margin-bottom: 10px;">
                    <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
                    <path d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 6.588zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z"/>
                </svg>
                <br>
                <strong>👆 Clicca sopra per caricare un'immagine</strong><br>
                Supportati formati: JPG, JPEG, PNG
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Colonna di output
    with col2:
        st.markdown('<p class="section-title">2. Analisi</p>', unsafe_allow_html=True)
        
        # Contenitore per il risultato con sfondo più chiaro per la leggibilità
        st.markdown('<div class="panel" style="background-color: #ffffff;">', unsafe_allow_html=True)
        
        # Bottone più grande e centrato per avviare l'analisi
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            analyze_button = st.button("🔍 Analizza immagine")
        
        if analyze_button and uploaded_image is not None:
            with st.spinner("Analisi in corso... Ricerca volti e confronto con le codifiche note..."):
                if isinstance(uploaded_image, str):
                    # Se è l'immagine demo
                    image = face_recognition.load_image_file(uploaded_image)
                else:
                    # Se è un'immagine caricata
                    image_array = np.array(uploaded_image)
                    image = image_array
                
                # Impostazione di una barra di progresso simulata
                progress_bar = st.progress(0)
                for percent_complete in range(101):
                    # Aggiornamento della barra di progresso più lento sui passaggi che sembrano più importanti
                    if percent_complete < 30:
                        time.sleep(0.01)  # Ricerca volti
                    elif percent_complete < 70:
                        time.sleep(0.02)  # Analisi codifiche
                    else:
                        time.sleep(0.005)  # Finalizzazione
                    progress_bar.progress(percent_complete)
                
                # Esegui il riconoscimento
                result_image, meloni_faces, result_message = recognize_meloni(image, meloni_encodings, tolerance)
                
                # Mostra il risultato con colori più intensi per migliorare la leggibilità
                if meloni_faces:
                    st.markdown(f"""
                    <div style="background-color: #d4edda; border-left: 6px solid #28a745; color: #155724; 
                    padding: 16px; border-radius: 4px; margin-bottom: 16px; font-size: 18px; font-weight: bold;">
                    ✅ {result_message}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Statistiche sul riconoscimento con colori più intensi
                    if len(meloni_faces) > 0:
                        max_confidence = max([face["confidence"] for face in meloni_faces]) * 100
                        
                        # Box per la metrica più visibile
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; color: #212529; padding: 15px; 
                        border-radius: 8px; margin-bottom: 20px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 16px; font-weight: 500; margin-bottom: 8px;">Confidenza massima</div>
                            <div style="font-size: 28px; font-weight: 700; color: #28a745;">{max_confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #fff3cd; border-left: 6px solid #ffc107; color: #856404; 
                    padding: 16px; border-radius: 4px; margin-bottom: 16px; font-size: 18px; font-weight: bold;">
                    ❌ {result_message}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostra l'immagine risultante
                st.image(result_image, caption="Risultato analisi", use_container_width=True)
                
                # Salva l'immagine
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                result_filename = f"risultato_{timestamp}.jpg"
                result_image.save(result_filename)
                
                # Aggiunta stile per rendere più evidente il pulsante di download
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; 
                margin-top: 20px; border: 1px dashed #6c757d;">
                """, unsafe_allow_html=True)
                
                # Aggiungi opzione per download con testo migliorato
                with open(result_filename, "rb") as file:
                    btn = st.download_button(
                        label="⬇️ Scarica immagine risultato",
                        data=file,
                        file_name=result_filename,
                        mime="image/jpeg"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Messaggio più dettagliato e visibile per guidare l'utente
            if uploaded_image is None and not use_demo:
                st.markdown("""
                <div style="background-color: #f8f9fa; border-left: 6px solid #6c757d; color: #495057; 
                padding: 16px; border-radius: 4px; margin-bottom: 16px; font-size: 16px;">
                <strong>📋 Passo 1:</strong> Prima carica un'immagine usando il selettore nella colonna a sinistra<br>
                <strong>📋 Passo 2:</strong> Poi premi il pulsante 'Analizza immagine' per iniziare il riconoscimento
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #e6f7ff; border-left: 6px solid #1890ff; color: #0c5460; 
                padding: 16px; border-radius: 4px; margin-bottom: 16px; font-size: 16px;">
                <strong>👆 Premi il pulsante 'Analizza immagine'</strong> per iniziare la ricerca di Meloni nell'immagine.<br>
                L'algoritmo cercherà volti e li confronterà con le immagini di riferimento nella cartella 'meloni_images'.
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Legenda colorata con maggior contrasto per la leggibilità
        st.markdown('<p class="section-title">Legenda</p>', unsafe_allow_html=True)
        
        # Pannello bianco per la legenda con testo scuro per massima leggibilità
        st.markdown("""
        <div class="panel" style="background-color: #ffffff; padding: 15px;">
            <div style="display: flex; align-items: center; margin-bottom: 15px; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
                <div style="width: 25px; height: 25px; background-color: #00FF00; margin-right: 15px; border: 1px solid #999;"></div>
                <div style="color: #333; font-weight: bold; font-size: 16px;">Riquadro <span style="color: #28a745;">VERDE</span>: Meloni identificata</div>
            </div>
            <div style="display: flex; align-items: center; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
                <div style="width: 25px; height: 25px; background-color: #FF0000; margin-right: 15px; border: 1px solid #999;"></div>
                <div style="color: #333; font-weight: bold; font-size: 16px;">Riquadro <span style="color: #dc3545;">ROSSO</span>: Altra persona</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()