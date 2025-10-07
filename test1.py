import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame.")
        break

    try:
        # Analyze emotions only
        predictions = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        # Handle both dict and list return formats (DeepFace >=0.0.79 returns list)
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Get dominant emotion
        dominant_emotion = predictions['dominant_emotion']

        # Get face region (x, y, w, h)
        region = predictions['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Put text of dominant emotion
        cv2.putText(
            frame,
            dominant_emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
        
        )

    except Exception as e:
        # If no face is found or error occurs, just skip
        pass

    # Show webcam
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quiting..")
        break

cap.release()
cv2.destroyAllWindows()
