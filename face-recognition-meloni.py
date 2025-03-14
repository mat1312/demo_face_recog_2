import face_recognition
import cv2
import os
import numpy as np

def create_meloni_detector(reference_images_paths):
    """Crea un rilevatore per Giorgia Meloni basato su immagini di riferimento locali."""
    known_meloni_encodings = []
    
    print("Caricamento delle immagini di riferimento di Giorgia Meloni...")
    
    # Processa ogni immagine di riferimento
    for i, image_path in enumerate(reference_images_paths):
        try:
            print(f"Elaborazione immagine di riferimento {i+1}: {image_path}")
            
            # Carica l'immagine
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Impossibile caricare l'immagine {image_path}. Verifica che il percorso sia corretto.")
                continue
            
            # Face recognition funziona con RGB, ma OpenCV usa BGR
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Trova i volti nell'immagine
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                print(f"  Nessun volto trovato nell'immagine {image_path}. Prova con un'altra immagine.")
                continue
            
            # Prendi il primo volto trovato (assumiamo sia Meloni)
            encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if encodings:
                known_meloni_encodings.append(encodings[0])
                print(f"  Encoding facciale estratto con successo dall'immagine {i+1}.")
            else:
                print(f"  Impossibile estrarre l'encoding dall'immagine {i+1}.")
        
        except Exception as e:
            print(f"Errore durante l'elaborazione dell'immagine {image_path}: {e}")
    
    if not known_meloni_encodings:
        raise ValueError("Impossibile estrarre encodings facciali dalle immagini di riferimento")
    
    print(f"Creato rilevatore con {len(known_meloni_encodings)} encodings di riferimento.")
    return known_meloni_encodings

def detect_meloni(image_path, known_meloni_encodings, tolerance=0.6):
    """
    Rileva Giorgia Meloni in un'immagine.
    
    Args:
        image_path: Percorso all'immagine locale
        known_meloni_encodings: Lista di encodings facciali di riferimento
        tolerance: Soglia di corrispondenza (più basso = più preciso)
        
    Returns:
        Immagine con box intorno ai volti di Meloni
    """
    # Carica l'immagine
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Impossibile caricare l'immagine {image_path}. Verifica che il percorso sia corretto.")
            return None
    except Exception as e:
        print(f"Errore durante il caricamento dell'immagine: {e}")
        return None
    
    # Converti in RGB (face_recognition usa RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Trova tutti i volti nell'immagine
    print("Ricerca volti nell'immagine...")
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    print(f"Trovati {len(face_locations)} volti.")
    
    if not face_locations:
        return image
    
    # Calcola gli encodings per ogni volto trovato
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    # Confronta ogni volto con gli encodings di Meloni
    for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
        # Controlla se il volto corrisponde a Meloni
        matches = face_recognition.compare_faces(known_meloni_encodings, face_encoding, tolerance=tolerance)
        
        # Se almeno un riferimento corrisponde, è probabilmente Meloni
        if True in matches:
            print(f"Identificata Giorgia Meloni nel volto {i+1}!")
            # Estrai le coordinate
            top, right, bottom, left = face_location
            
            # Disegna un rettangolo
            color = (0, 0, 255)  # Rosso in BGR
            thickness = 2
            cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
            
            # Aggiungi etichetta
            label = "Giorgia Meloni"
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, label, (left, top - 10), font, 0.7, color, thickness)
        else:
            print(f"Il volto {i+1} non è Giorgia Meloni.")
    
    return image

def main():
    # Definisci i percorsi delle immagini di riferimento di Giorgia Meloni
    # Sostituisci questi percorsi con quelli delle tue immagini locali
    reference_images = [
        "reference_images\meloni1.jpg",
        "reference_images\meloni2.jpg",
        "reference_images\meloni3.jpg",
        "reference_images\meloni4.jpg",
        "reference_images\meloni5.jpg",
        "reference_images\meloni6.jpg",
        "reference_images\meloni.jpg",
    ]
    
    # Crea il rilevatore di Meloni
    try:
        known_meloni_encodings = create_meloni_detector(reference_images)
    except Exception as e:
        print(f"Errore nella creazione del rilevatore: {e}")
        return
    
    # Test su un'immagine (sostituisci con il percorso della tua immagine di test)
    test_image = "test_image\meloni_in_gruppo.jpg"
    
    print(f"\nTestando il riconoscimento su: {test_image}")
    
    # Rileva Meloni nell'immagine
    result_image = detect_meloni(test_image, known_meloni_encodings)
    
    if result_image is not None:
        # Salva il risultato
        output_path = "risultato_meloni.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"\nRisultato salvato in: {output_path}")
        
        # Mostra l'immagine (opzionale, richiede ambiente grafico)
        try:
            cv2.imshow("Rilevamento Giorgia Meloni", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Impossibile visualizzare l'immagine (potrebbe mancare un ambiente grafico).")
    else:
        print("Elaborazione dell'immagine fallita.")

if __name__ == "__main__":
    main()
