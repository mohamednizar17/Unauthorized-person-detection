Unauthorized Entry Detection System
Overview
This project implements a real-time face recognition system to detect authorized and unauthorized individuals using a webcam. It uses FaceNet for face recognition, MTCNN for face detection, and OpenCV for webcam processing. The system is designed to recognize faces shown on a mobile phone screen, labeling authorized faces in green and unauthorized faces in red in an OpenCV window. It includes preprocessing to enhance accuracy and logs detections to a file and image captures.
Features

Real-Time Face Recognition: Identifies faces as "Authorized" or "Unauthorized" with confidence scores.
MTCNN Face Detection: Robust detection for mobile phone images.
Preprocessing: Enhances and resizes images for optimal FaceNet performance.
Real-Time Feedback: Displays face count, status, and detection log in the console and OpenCV window.
Logging: Saves unauthorized detection frames to logs/ and logs all detections to security_log.txt.
No External Services: Runs locally in VS Code with python app.py, no email or web dependencies.

Project Structure
unauthorized-entry-detection/
├── authorized_person/              # Original face images (e.g., John_Doe/image1.jpg)
├── preprocessed_authorized_person/ # Preprocessed face images (optional)
├── logs/                          # Saved frames for unauthorized detections
├── authorized_encodings.pkl       # FaceNet embeddings from training
├── app.py                        # Main script for real-time face recognition
├── train_face_recognition_facenet.py # Training script to generate embeddings
├── preprocess_faces.py           # Preprocessing script for training images
├── security_log.txt              # Log file for detections
├── requirements.txt              # Python dependencies
├── README.md                     # This file

Prerequisites

Python 3.11: Ensure Python is installed.
Webcam: A high-resolution webcam (720p+ recommended).
Training Data: Over 300 MB of face images in authorized_person/ (subfolders named by person, e.g., John_Doe/image1.jpg).
VS Code: For running and debugging scripts.

Installation

Clone or Set Up the Project Directory:

Ensure the project is in C:\Users\arniz\OneDrive\Desktop\unauthorized-entry-detection.
If using OneDrive causes sync issues, move to a local directory:move c:\Users\arniz\OneDrive\Desktop\unauthorized-entry-detection c:\Users\arniz\Desktop\unauthorized-entry-detection
cd c:\Users\arniz\Desktop\unauthorized-entry-detection




Install Dependencies:

Install required packages from requirements.txt:pip install -r requirements.txt




Prepare Training Data:

Place face images in authorized_person/<person_name>/*.jpg (or .jpeg, .png).
Each person should have 10–20 clear, frontal face images with varied lighting and angles.



Usage

Preprocess Training Data (recommended for better accuracy):

Run the preprocessing script to standardize images:python preprocess_faces.py


This creates preprocessed_authorized_person/ with cropped, resized (160x160), and enhanced images.
Check security_log.txt for any failed images and fix them (e.g., remove images with no/single face).


Train the Model:

Update train_face_recognition_facenet.py to use preprocessed images (if used):AUTHORIZED_DIR = "preprocessed_authorized_person"


Run the training script to generate authorized_encodings.pkl:python train_face_recognition_facenet.py




Run the Application:

Open VS Code in the project directory:cd c:\Users\arniz\OneDrive\Desktop\unauthorized-entry-detection
code .


Run the main script:python app.py


An OpenCV window will display the webcam feed with bounding boxes (green for authorized, red for unauthorized) and status text.
The console shows a table of the last 10 detections (timestamp, name, status, confidence).
Press q to quit.


Testing:

Show clear, frontal face images on a mobile phone to the webcam (30–50 cm away, bright screen).
Authorized faces are labeled with names and confidence scores (e.g., “John Doe (0.85)”).
Unauthorized faces are labeled “Unauthorized” and saved to logs/.



Tuning for Accuracy

Threshold Adjustment:
Edit FACENET_THRESHOLD in app.py (default 0.4):FACENET_THRESHOLD = 0.3  # Lower for more matches, higher for stricter


Check confidence scores in the console to find an optimal value.


Training Data:
Ensure 10–20 images per person with varied lighting/angles.
Use preprocess_faces.py to standardize images.


Webcam Setup:
Use a 720p+ webcam and bright mobile phone screen.
Position the phone to avoid pixelation.



Troubleshooting

No Faces Detected:

Test the webcam:import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


Ensure mobile phone images are clear and frontal.
In app.py, set enforce_detection=False in is_authorized_face for less strict detection (may reduce accuracy).


Recognition Errors:

Check security_log.txt for errors.
Print similarity scores in is_authorized_face:logging.info(f"Similarity scores: {distances}")


Adjust FACENET_THRESHOLD in app.py.


Performance Issues:

Increase time.sleep(0.03) to 0.05 in app.py if lagging:time.sleep(0.05)


Use a GPU for faster MTCNN/FaceNet processing.


Missing Encodings:

Rerun train_face_recognition_facenet.py if authorized_encodings.pkl is missing.



Notes

The system uses FaceNet for recognition and MTCNN for detection, optimized for mobile phone images.
Logs are saved to security_log.txt and unauthorized frames to logs/.
No internet or external services are required.

Contact
For issues or enhancements, please provide:

security_log.txt output.
Details of your dataset (number of persons, images per person).
Specific recognition errors or test case descriptions.
