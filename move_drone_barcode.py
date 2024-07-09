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

# Helper function to detect barcode in an image
def detect_barcode(image):
    barcodes = pyzbar.decode(image)
    return barcodes

while True:
    # Capture frame from drone
    frame = drone.get_frame_read().frame
    
    # Run the model with confidence threshold
    output = model(frame, conf=CONFIDENCE_THRESHOLD)
    
    # Extract boxes from the model's output
    boxes = output[0].boxes
    
    # Annotate the frame with bounding boxes and labels
    for box in boxes:
        # Convert tensors to Python scalars
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Move the drone to align with the box center
        frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
        error_x = center_x - frame_center_x
        error_y = center_y - frame_center_y
        
        if abs(error_x) > 20:  # Adjust this threshold as needed
            if error_x > 0:
                drone.move_right(20)
            else:
                drone.move_left(20)
                
        if abs(error_y) > 20:  # Adjust this threshold as needed
            if error_y > 0:
                drone.move_down(20)
            else:
                drone.move_up(20)
                
        if abs(error_x) <= 20 and abs(error_y) <= 20:
            drone.move_forward(20)
            
        # If the drone is close enough to the box, break and look for barcode
        if x2 - x1 > frame.shape[1] * 0.3:  # Adjust this threshold as needed
            drone.move_forward(20)
            barcodes = detect_barcode(frame)
            if barcodes:
                # Assuming only one barcode is detected
                barcode = barcodes[0]
                (bx, by, bw, bh) = barcode.rect
                barcode_img = frame[by:by+bh, bx:bx+bw]
                cv2.imwrite("barcode.jpg", barcode_img)
                break
    
    # Display the annotated frame
    cv2.imshow('Video', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
drone.streamoff()
