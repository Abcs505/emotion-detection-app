import cv2
from fer import FER

# Initialize the FER emotion detector with MTCNN face detector (more accurate)
detector = FER(mtcnn=True)

# Open webcam (0 is usually default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions on the current frame
    results = detector.detect_emotions(frame)

    # Draw boxes and labels for all detected faces
    for face in results:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]

        # Get dominant emotion with highest score
        dominant_emotion = max(emotions, key=emotions.get)
        score = emotions[dominant_emotion]

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display emotion and confidence score
        text = f"{dominant_emotion} ({score:.2f})"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show the video frame with annotations
    cv2.imshow("Emotion Detection - FER", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
