import cv2
import numpy as np
from djitellopy import Tello

drone = Tello()  # declaring drone object
drone.connect()
drone.streamon()  # start camera streaming

while True:
    frame = drone.get_frame_read().frame  # capturing frame from drone

    cv2.imshow('Video', frame)  # show corrected frame on the display

    if cv2.waitKey(1) & 0xFF == ord('q'):  # quit from script
        break

cv2.destroyAllWindows()
drone.streamoff()