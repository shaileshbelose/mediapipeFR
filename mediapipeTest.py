import cv2
import mediapipe as mp
import time
from deepface import DeepFace

# Placeholder function for spoof detection (you would need to implement this)
def is_spoof(face_image):
    try:
        # Analyze emotions and other attributes
        result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        print(result)
        # Get confidence and dominant emotion
        emotion_confidence = result[0].get("face_confidence", 0.0)
        dominant_emotion = result[0].get("dominant_emotion", "")

        # Check for low confidence or inconsistent dominant emotion as an indicator of spoofing
        if emotion_confidence < 0.85 or dominant_emotion == "neutral":
            return True
        return False
    except Exception as e:
        print("DeepFace error:", e)
        # If detection fails, mark as possible spoof
        return True

cap = cv2.VideoCapture("5.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # Extract face region for spoof detection
            face_image = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            if face_image.size == 0:
                continue  # Skip if bounding box is invalid

            # Check if face is spoofed
            spoofed = is_spoof(face_image)

            # Draw bounding box with color based on spoof detection result
            color = (0, 0, 255) if spoofed else (0, 255, 0)  # Red for spoofed, Green for real
            cv2.rectangle(img, bbox, color, 2)
            
            # Display confidence score
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), 
                        cv2.FONT_HERSHEY_PLAIN, 3, color, 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # Display the frame rate (optional)
    # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break  # Press 'q' to exit

cap.release()
cv2.destroyAllWindows()
