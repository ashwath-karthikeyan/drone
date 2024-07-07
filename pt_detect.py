from ultralytics import YOLO
import cv2

# Paths to the model weights and the image
MODEL_WEIGHT_PATH = 'weights.onnx'
# IMAGE_PATH = 'image.jpg'
IMAGE_PATH = 'box_real.jpeg'

# Initialize the YOLO model with the converted model
model = YOLO(MODEL_WEIGHT_PATH, task='detect')

# Load and preprocess the image using cv2
image = cv2.imread(IMAGE_PATH)

# Run the model
output = model(image)

# Extract boxes from the model's output
boxes = output[0].boxes

# Annotate the image with bounding boxes and labels
for box in boxes:
    # Convert tensors to Python scalars
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the annotated image
output_image_path = 'annotated_image.jpg'
cv2.imwrite(output_image_path, image)