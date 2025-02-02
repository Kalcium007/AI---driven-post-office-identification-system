import cv2
import pytesseract
from googletrans import Translator
import re

# Set up Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Google Translator
translator = Translator()

# Open the camera
cap = cv2.VideoCapture(0)

# Regular expression to check for Tamil characters
tamil_characters = re.compile("[\u0B80-\u0BFF]")

if not cap.isOpened():
    print("Error: Could not open the webcam.")
else:
    print("Press 'c' to capture the image or 'q' to quit.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read a frame from the webcam.")
            break

        # Show the live feed
        cv2.imshow('Live Feed - Press "c" to Capture, "q" to Quit', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # If 'c' key is pressed, capture the image and process it
        if key == ord('c'):
            # OCR with original color frame
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(frame, lang='tam+eng', config=custom_config)

            # Check if the text contains Tamil characters
            if tamil_characters.search(text):
                # Translate to English if Tamil text is detected
                translated_text = translator.translate(text, dest='en').text
                print("Recognized Text (Translated to English):\n", translated_text)
            else:
                # Just display the text if it's in English
                print("Recognized Text (English):\n", text)

            # Display the captured frame with recognized text
            cv2.imshow('Captured Frame', frame)
            cv2.waitKey(0)  # Wait for a key press to close the displayed frame

        # If 'q' key is pressed, quit the loop
        elif key == ord('q'):
            print("Exiting...")
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
