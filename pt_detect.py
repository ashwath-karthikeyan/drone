from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Paths to the model weights and the image
MODEL_WEIGHT_PATH = '1/weights.onnx'
IMAGE_PATH = 'image.jpg'
# IMAGE_PATH = 'box.jpg'

# Initialize the YOLO model with the converted model
model = YOLO(MODEL_WEIGHT_PATH, task='detect')

# Load and preprocess the image using cv2
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(image_rgb, (640, 640))  # Resize image to match model's expected input size
input_image = np.transpose(resized_image, (2, 0, 1))  # Change data layout from HWC to CHW
input_image = input_image.astype(np.float32) / 255.0  # Normalize the image to [0, 1]
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# Convert to PyTorch tensor
input_tensor = torch.tensor(input_image)

# Run the model
with torch.no_grad():
    output = model(input_tensor)

# Print the model's output
print(output)