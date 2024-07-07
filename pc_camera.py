import cv2
import numpy as np
from ultralytics import YOLO

# Paths to the model weights
MODEL_WEIGHT_PATH = 'weights.onnx'
CONFIDENCE_THRESHOLD = 0.8  # Set your desired confidence threshold here

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the YOLO model
model = YOLO(MODEL_WEIGHT_PATH, task='detect')

while True:
    # Capture frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run the model with confidence threshold
    output = model(frame, conf=CONFIDENCE_THRESHOLD)
    
    # Extract boxes from the model's output
    boxes = output[0].boxes
    
    # Annotate the frame with bounding boxes and labels
    for box in boxes:
        # Convert tensors to Python scalars
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the annotated frame
    cv2.imshow('Video', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
