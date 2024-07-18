import cv2
from pyzbar import pyzbar

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the ID for the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def detect_barcode(frame):
    barcodes = pyzbar.decode(frame)
    return barcodes

while True:
    ret, frame = cap.read()  # Capture frame from webcam

    if not ret:
        print("Error: Could not read frame.")
        break

    barcodes = detect_barcode(frame)
    if barcodes:
        for barcode in barcodes:
            print("Barcode detected:", barcode.data.decode('utf-8'))
        break

    cv2.imshow('Video', frame)  # Show the frame on the display

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit the script when 'q' is pressed
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows