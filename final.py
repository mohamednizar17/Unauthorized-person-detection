import cv2
import numpy as np
import time
from datetime import datetime
import os
import pickle
from deepface import DeepFace
import logging

# Setup logging
LOG_FILE = "security_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Face recognition setup
ENCODINGS_FILE = "authorized_encodings.pkl"
FACENET_THRESHOLD = 0.4  # Adjustable threshold for FaceNet similarity

def load_known_faces():
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['embeddings'], data['names']
    except FileNotFoundError:
        print("Encodings file not found. Please run training script first.")
        logging.error("Encodings file not found.")
        return [], []

def preprocess_face(face_img):
    """Preprocess face image: enhance and resize to 160x160."""
    try:
        # Brightness/contrast adjustment
        face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)
        # Resize to FaceNet input size
        face_img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)
        return face_img
    except Exception as e:
        logging.error(f"Error preprocessing face: {e}")
        return None

def is_authorized_face(face_img, known_embeddings, known_names, threshold):
    if not known_embeddings:
        return False, "Unknown", 0.0
    try:
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_img)
        
        embedding = DeepFace.represent(temp_path, model_name="Facenet", detector_backend="mtcnn", enforce_detection=True)
        if not embedding:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, "Unknown", 0.0
        
        embedding = np.array(embedding[0]["embedding"])
        
        # Compute cosine similarity
        distances = [np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb)) 
                     for known_emb in known_embeddings]
        max_similarity = max(distances) if distances else -1
        
        if max_similarity > threshold:
            match_index = distances.index(max_similarity)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return True, known_names[match_index], max_similarity
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, "Unknown", max_similarity
    except Exception as e:
        logging.error(f"Face recognition error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, "Unknown", 0.0

def process_frame(frame, known_embeddings, known_names, threshold, detection_log):
    # Detect faces using MTCNN
    try:
        faces = DeepFace.extract_faces(frame, detector_backend="mtcnn", enforce_detection=False)
    except Exception as e:
        logging.warning(f"Face detection error: {e}")
        faces = []

    face_count = len(faces)
    any_unauthorized = False
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for face_data in faces:
        facial_area = face_data["facial_area"]
        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
        
        # Extract and preprocess face
        face_img = frame[y:y+h, x:x+w]
        face_img = preprocess_face(face_img)
        if face_img is None:
            continue

        # Recognize face
        is_authorized, person_name, confidence = is_authorized_face(face_img, known_embeddings, known_names, threshold)
        
        # Update detection log
        detection_log.append({
            "Timestamp": timestamp,
            "Name": person_name,
            "Status": "Authorized" if is_authorized else "Unauthorized",
            "Confidence": f"{confidence:.2f}"
        })
        if len(detection_log) > 10:  # Keep last 10 detections
            detection_log.pop(0)

        # Draw rectangle and label
        color = (0, 255, 0) if is_authorized else (0, 0, 255)  # Green for authorized, red for unauthorized
        message = f"{person_name} ({confidence:.2f})" if is_authorized else "Unauthorized"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if not is_authorized:
            any_unauthorized = True
            logging.info(f"Unauthorized face detected at {timestamp}")

    # Log frame if unauthorized
    if any_unauthorized:
        os.makedirs("logs", exist_ok=True)
        cv2.imwrite(f"logs/frame_{timestamp.replace(':', '-')}.jpg", frame)
    
    # Display status on frame
    status = f"{'ALERT: Unauthorized' if any_unauthorized else 'Authorized/No Faces'} ({face_count} faces)"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Print detection log to console
    if detection_log:
        print("\nRecent Detections (Last 10):")
        print(f"{'Timestamp':<20} {'Name':<15} {'Status':<12} {'Confidence':<10}")
        print("-" * 60)
        for entry in detection_log:
            print(f"{entry['Timestamp']:<20} {entry['Name']:<15} {entry['Status']:<12} {entry['Confidence']:<10}")
    
    return frame

def main():
    # Load known faces
    known_embeddings, known_names = load_known_faces()
    if not known_embeddings:
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        logging.error("Could not open video capture")
        return

    detection_log = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                logging.error("Failed to capture frame")
                break

            # Process frame
            frame = process_frame(frame, known_embeddings, known_names, FACENET_THRESHOLD, detection_log)

            # Display frame
            cv2.imshow("Face Recognition System", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.03)  # Approx 30 FPS

    except Exception as e:
        print(f"Error in main loop: {e}")
        logging.error(f"Error in main loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()