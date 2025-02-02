import cv2
import easyocr
from deep_translator import GoogleTranslator

# Initialize EasyOCR with English and Tamil
try:
    reader = easyocr.Reader(['en'], gpu=False)
except RuntimeError:
    print("There was an issue loading the language model. Try updating EasyOCR or re-installing.")

# Initialize the translator for English translation
translator = GoogleTranslator(source='auto', target='en')

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Display the live camera feed
    cv2.imshow('Live Feed - Press "c" to Capture and Recognize Text', frame)

    # Wait for the key press
    key = cv2.waitKey(1) & 0xFF

    # Check if the 'c' key was pressed for capturing
    if key == ord('c'):
        # Detect and recognize text in the captured frame
        try:
            result = reader.readtext(frame)
        except Exception as e:
            print("Error recognizing text:", e)
            continue

        # Collect recognized text in a list
        text_paragraph = []

        # Draw bounding boxes and collect text
        for detection in result:
            top_left = tuple(map(int, detection[0][0]))  # Convert top-left coordinates to integers
            bottom_right = tuple(map(int, detection[0][2]))  # Convert bottom-right coordinates to integers
            text = detection[1]

            # Add recognized text to the list
            text_paragraph.append(text)
            
            # Draw the bounding box
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            
            # Display the recognized text on the frame
            cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Join recognized words into a paragraph
        paragraph_output = " ".join(text_paragraph)

        # Translate the paragraph to English using deep-translator
        try:
            translated_text = translator.translate(paragraph_output)
            print("Recognized Text (Translated to English):\n", translated_text)
        except Exception as e:
            print("Error translating text:", e)

        # Display the captured frame with recognized text
        cv2.imshow('Captured Text', frame)

    # Press 'q' to exit the loop
    elif key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
