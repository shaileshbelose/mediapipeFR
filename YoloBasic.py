import cv2
from ultralytics import YOLO

# Load the pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Perform inference on the image
results = model("33.png")  # Run detection

# Loop through the results and display them
for result in results:
    # result.plot() returns a numpy array with bounding boxes drawn on it
    detected_image = result.plot()
    
    # Show the detected image using OpenCV
    cv2.imshow("Detections", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
