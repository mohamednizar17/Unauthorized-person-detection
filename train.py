import os
import pickle
import numpy as np
from deepface import DeepFace
import cv2

# Directory containing authorized persons' images
AUTHORIZED_DIR = "preprocessed_authorized_persons_resized"
ENCODINGS_FILE = "authorized_encodings.pkl"

def load_and_encode_faces():
    known_embeddings = []
    known_names = []

    # Check if AUTHORIZED_DIR exists
    if not os.path.exists(AUTHORIZED_DIR):
        print(f"Error: Directory '{AUTHORIZED_DIR}' does not exist.")
        return [], []

    # Iterate through each person's subfolder
    for person_name in os.listdir(AUTHORIZED_DIR):
        person_dir = os.path.join(AUTHORIZED_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing images for {person_name}...")
        for img_name in os.listdir(person_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_dir, img_name)
                try:
                    # Load image with OpenCV for validation
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error: Failed to load {img_name}")
                        continue
                    
                    # Validate face presence using Haar Cascade
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    if len(faces) != 1:
                        print(f"Error: {img_name} has {len(faces)} faces (expected 1)")
                        continue

                    # Generate embedding using FaceNet
                    embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=True)
                    if embedding:
                        known_embeddings.append(np.array(embedding[0]["embedding"]))
                        known_names.append(person_name)
                        print(f"Encoded {img_name}")
                    else:
                        print(f"No faces found in {img_name}")
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

    return known_embeddings, known_names

def save_encodings(embeddings, names):
    # Save embeddings and names to a pickle file
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'names': names}, f)
    print(f"Saved embeddings to {ENCODINGS_FILE}")

def main():
    # Load and encode faces
    embeddings, names = load_and_encode_faces()
    
    if not embeddings:
        print("No valid face embeddings found. Exiting.")
        return
    
    # Save embeddings
    save_encodings(embeddings, names)
    print(f"Training complete. Encoded {len(embeddings)} faces for {len(set(names))} persons.")

if __name__ == "__main__":
    main()