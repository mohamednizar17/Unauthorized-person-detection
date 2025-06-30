import os
import cv2
import numpy as np
from PIL import Image
import logging

# Input and output directories
INPUT_DIR = "authorized_persons"
OUTPUT_DIR = "preprocessed_authorized_persons_resized"
LOG_FILE = "preprocess_log.txt"

# Face detection setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Failed to load haarcascade_frontalface_default.xml")
    exit(1)

def setup_logging():
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

def preprocess_image(img_path, output_path):
    """Preprocess a single image: detect face, crop, resize, and enhance."""
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Failed to load image: {img_path}")
            return False

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
        if len(faces) != 1:
            logging.warning(f"Invalid number of faces ({len(faces)}) in {img_path}")
            return False

        # Get face coordinates
        (x, y, w, h) = faces[0]

        # Add padding to include more of the face (e.g., forehead, chin)
        padding = int(max(w, h) * 0.2)
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, img.shape[1])
        h = min(h + 2 * padding, img.shape[0])

        # Crop face
        face_img = img[y:y+h, x:x+w]

        # Enhance image (brightness/contrast adjustment)
        face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)

        # Resize to 160x160 (FaceNet input size)
        face_img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)

        # Save preprocessed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, face_img)
        logging.info(f"Preprocessed and saved: {output_path}")
        return True

    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return False

def preprocess_dataset():
    """Preprocess all images in the authorized_person directory."""
    setup_logging()
    logging.info("Starting preprocessing...")

    if not os.path.exists(INPUT_DIR):
        logging.error(f"Input directory '{INPUT_DIR}' does not exist.")
        return

    processed_count = 0
    failed_count = 0

    # Iterate through each person's subfolder
    for person_name in os.listdir(INPUT_DIR):
        person_dir = os.path.join(INPUT_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        output_person_dir = os.path.join(OUTPUT_DIR, person_name)
        for img_name in os.listdir(person_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_dir, img_name)
                output_path = os.path.join(output_person_dir, img_name)
                if preprocess_image(img_path, output_path):
                    processed_count += 1
                else:
                    failed_count += 1

    logging.info(f"Preprocessing complete: {processed_count} images processed, {failed_count} failed.")
    print(f"Preprocessed images saved to '{OUTPUT_DIR}'. Check '{LOG_FILE}' for details.")

def main():
    preprocess_dataset()

if __name__ == "__main__":
    main()