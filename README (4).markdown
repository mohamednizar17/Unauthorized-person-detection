# Unauthorized Person Detection System

## Overview
This repository contains two Python-based models for detecting unauthorized persons using computer vision techniques:

1. **Face Recognition Model (`final.py`)**: Utilizes the DeepFace library with FaceNet to distinguish between authorized and unauthorized faces by comparing facial embeddings against a pre-trained dataset.
2. **Restricted Area Detection Model (`final1.py`)**: Detects faces in a user-defined, draggable, and resizable restricted area using Haar Cascade, flagging any detected face as an intruder.

## Prerequisites
- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- DeepFace (for `final.py`)
- A webcam for real-time video capture
- Pre-trained face encodings file (`authorized_encodings.pkl`) for `final.py`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mohamednizar17/Unauthorized-person-detection.git
   cd Unauthorized-person-detection
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python numpy deepface
   ```
3. For `final.py`, ensure the `authorized_encodings.pkl` file is available. If not, run the training script to generate face encodings (refer to the working guide for details).
4. Ensure OpenCV's Haar Cascade file (`haarcascade_frontalface_default.xml`) is accessible (included with OpenCV).

## Usage
### Face Recognition Model (`final.py`)
- **Purpose**: Identifies authorized vs. unauthorized individuals using facial recognition.
- **Run the script**:
  ```bash
  python final.py
  ```
- **Operation**:
  - The webcam captures video frames.
  - Faces are detected and compared against known embeddings.
  - Authorized faces are marked with green rectangles, unauthorized with red.
  - Logs unauthorized detections and saves frames to the `logs/` directory.
  - Press `q` to exit.

### Restricted Area Detection Model (`final1.py`)
- **Purpose**: Detects faces in a user-defined restricted area, flagging them as intruders.
- **Run the script**:
  ```bash
  python final1.py
  ```
- **Operation**:
  - A green square (restricted area) is displayed, which turns red if a face is detected inside.
  - Drag the square with the mouse to reposition it.
  - Press `q` to exit.

## Directory Structure
- `final.py`: Face recognition model script.
- `final1.py`: Restricted area detection model script.
- `authorized_encodings.pkl`: Pre-trained face encodings (required for `final.py`).
- `logs/`: Directory for storing unauthorized detection frames (auto-created).
- `security_log.txt`: Log file for face recognition events.

## Notes
- Ensure a webcam is connected and accessible.
- For `final.py`, the `authorized_encodings.pkl` file must be generated beforehand using a training script.
- The restricted area in `final1.py` is draggable but not resizable in the current implementation. Future updates may include resizing functionality.
- Logs and frames are saved for unauthorized detections to aid in security monitoring.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs or feature requests.

## License
This project is licensed under the MIT License.