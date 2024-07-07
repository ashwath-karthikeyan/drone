from ultralytics import YOLO
import cv2

# Paths to the model weights and the image
MODEL_WEIGHT_PATH = 'weights.onnx'
IMAGE_PATH = 'image.jpg'
# IMAGE_PATH = 'box.jpg'

# Initialize the YOLO model with the converted model
model = YOLO(MODEL_WEIGHT_PATH, task='detect')

# Load and preprocess the image using cv2
image = cv2.imread(IMAGE_PATH)

# Run the model
output = model(image)

# Print the model's output
boxes = (output[0].boxes)