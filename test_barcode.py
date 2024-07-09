from pyzbar import pyzbar
import cv2

def detect_barcode(image_path):
    image = cv2.imread(image_path)
    barcodes = pyzbar.decode(image)
    return barcodes

barcodes = detect_barcode('barcode.png')
for barcode in barcodes:
    print(barcode.data.decode('utf-8'))