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

def detect_barcode(image):
    barcodes = pyzbar.decode(image)
    return barcodes

barcode_detected = False
barcode_text = ""

while not barcode_detected:
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

    # Detect barcodes
    barcodes = detect_barcode(frame)
    if barcodes:
        barcode_detected = True
        barcode_text = barcodes[0].data.decode('utf-8')
    
    # Display the annotated frame
    cv2.imshow('Video', frame)
    
    # Break the loop on 'q' key press (for safety)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Display the barcode text
if barcode_detected:
    blank_image = np.zeros((500, 1000, 3), np.uint8)
    cv2.putText(blank_image, f'Barcode: {barcode_text}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Barcode Detected', blank_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
drone.streamoff()