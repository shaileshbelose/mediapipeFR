import os
import cv2
from ultralytics import YOLO

# Set environment variable to increase read attempts for FFMPEG
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

# Load the pretrained YOLOv8 model
model = YOLO('yolo11m-pose.pt')

# Open the video file
video_path = "Videos/9.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video frame width, height, and FPS for output settings
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object to save output
output_path = "output_detected.avi"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Warning: Skipping a frame due to read error.")
        continue  # Skip to the next frame if reading fails

    # Perform YOLOv8 detection on the frame
    results = model(frame)
    
    # Retrieve and draw detections on the frame
    for result in results:
        detected_frame = result.plot()  # Draw detections on the frame

    # Display the frame with detections
    cv2.imshow("YOLO11pose Detections", detected_frame)
    
    # Write the detected frame to the output video
    out.write(detected_frame)

    # Press 'q' to exit video display early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
