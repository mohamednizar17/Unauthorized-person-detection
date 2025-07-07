import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not load face cascade.")
    exit()

# Square properties
square_x, square_y = 100, 100
square_size = 200
square_color = (0, 255, 0)  # Green
is_dragging = False

# Mouse callback for dragging square
def mouse_event(event, x, y, flags, param):
    global square_x, square_y, is_dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        if square_x <= x <= square_x + square_size and square_y <= y <= square_y + square_size:
            is_dragging = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging:
            square_x, square_y = x - square_size // 2, y - square_size // 2
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False

# Set mouse callback
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_event)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Reset alert
    alert = False
    square_color = (0, 255, 0)  # Green by default

    # Check for faces in square
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw face rectangle
        if (square_x < x + w and square_x + square_size > x and
            square_y < y + h and square_y + square_size > y):
            alert = True
            square_color = (0, 0, 255)  # Red if face detected in square

    # Draw square
    cv2.rectangle(frame, (square_x, square_y),
                  (square_x + square_size, square_y + square_size),
                  square_color, 2)

    # Display alert if face detected in square
    if alert:
        cv2.putText(frame, "INTRUDER!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display instructions
    cv2.putText(frame, "Drag square, press 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()