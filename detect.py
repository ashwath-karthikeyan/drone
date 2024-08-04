import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
from pyzbar import pyzbar
import threading
from pynput import keyboard

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

def hover():
    drone.send_rc_control(0, 0, 0, 0)

def on_press(key):
    try:
        if key.char == 'w':
            print("Move forward")
            drone.move_forward(30)
        elif key.char == 'a':
            print("Move left")
            drone.move_left(30)
        elif key.char == 's':
            print("Move back")
            drone.move_back(30)
        elif key.char == 'd':
            print("Move right")
            drone.move_right(30)
        elif key.char == 't':
            print("Take off")
            drone.takeoff()
            threading.Timer(2.0, hover).start()  # Hover after 2 seconds
        elif key.char == 'l':
            print("Land")
            drone.land()
    except AttributeError:
        if key == keyboard.Key.up:
            print("Move up")
            drone.move_up(30)
        elif key == keyboard.Key.down:
            print("Move down")
            drone.move_down(30)
        elif key == keyboard.Key.left:
            print("Rotate counter clockwise")
            drone.rotate_counter_clockwise(30)
        elif key == keyboard.Key.right:
            print("Rotate clockwise")
            drone.rotate_clockwise(30)
        elif key == keyboard.Key.esc:
            return False

# Start the listener thread for keyboard input
listener = keyboard.Listener(on_press=on_press)
listener.start()

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

drone.streamoff()
drone.land()
cv2.destroyAllWindows()

# Ensure the listener thread ends
listener.stop()