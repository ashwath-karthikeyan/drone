import cv2

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the ID for the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame from webcam

    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow('Video', frame)  # Show the frame on the display

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit the script when 'q' is pressed
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows