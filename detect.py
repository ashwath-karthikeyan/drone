import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
from pyzbar import pyzbar

# Paths to the model weights
MODEL_WEIGHT_PATH = 'weights.onnx'
CONFIDENCE_THRESHOLD = 0.85 

# Initialize the drone
drone = Tello()
drone.connect()
drone.streamon()

# Initialize the YOLO model
model = YOLO(MODEL_WEIGHT_PATH, task='detect')

while True:
    # Capture frame from drone
    raw_frame = drone.get_frame_read().frame

    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    
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
drone.streamoff()