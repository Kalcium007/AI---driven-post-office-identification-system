import cv2
import easyocr
from deep_translator import GoogleTranslator
from transformers import pipeline
import requests
import json
import logging
import re  # For regular expression to detect PIN code
from PIL import Image

# -------------------------------
# Configuration and Setup
# -------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize EasyOCR with English
try:
    reader = easyocr.Reader(['en'], gpu=False)
except RuntimeError:
    logging.error("There was an issue loading the EasyOCR language model. Try updating or re-installing EasyOCR.")

# Initialize Translator
translator = GoogleTranslator(source='auto', target='en')

# Initialize NLP model for address parsing
address_parser = pipeline("ner", model="dslim/bert-base-NER")

# Postal Pincode API endpoint
API_ENDPOINT = "https://api.postalpincode.in/pincode/"

# -------------------------------
# Utility Functions
# -------------------------------

def preprocess_response(response_text):
    """Mock preprocessing to simulate ML-like response validation."""
    if not response_text.strip():
        return "EmptyResponse"
    try:
        data = json.loads(response_text)
        if not data or "PostOffice" not in data[0] or not data[0]["PostOffice"]:
            return "InvalidStructure"
        return "ValidResponse"
    except json.JSONDecodeError:
        return "ParseError"

def validate_pincode(pincode, scanned_text):
    """Validate the PIN code using an external API and compare with OCR scanned text."""
    endpoint = f"{API_ENDPOINT}{pincode}"
    try:
        response = requests.get(endpoint)
        logging.info("API request sent, awaiting response.")
        if response.status_code != 200:
            logging.error(f"Unable to fetch data. HTTP Status Code: {response.status_code}")
            return f"Error: HTTP Status Code {response.status_code}"

        response_status = preprocess_response(response.text)
        logging.info(f"Response status after preprocessing: {response_status}")

        if response_status == "EmptyResponse":
            return "Error: Empty response from the server."
        elif response_status == "ParseError":
            return "Error: Failed to parse JSON response."
        elif response_status == "InvalidStructure":
            return "Error: Invalid response structure or no data found for this PIN code."
        else:
            pincode_information = json.loads(response.text)
            necessary_information = pincode_information[0]["PostOffice"][0]
            region_name = necessary_information.get("Region", "").lower()
            district = necessary_information.get("District", "Unknown")
            state = necessary_information.get("State", "Unknown")
            branch_office = necessary_information.get("Name", "Unknown")
            
            # Check if any word in the scanned text matches the region name
            if any(word.lower() == region_name for word in scanned_text.split()):
                logging.info("Validation Successful: Address matches with scanned text.")
                return f"Validation Successful: Address is correct.\nBranch Office: {branch_office}\nDistrict: {district}\nState: {state}"
            else:
                logging.warning("Validation Failed: Address does not match scanned text.")
                return f"Validation Failed: Address is incorrect.\nCorrect Details:\nBranch Office: {branch_office}\nDistrict: {district}\nState: {state}"
    except requests.RequestException as e:
        logging.error(f"Network or API issue. {e}")
        return f"Error: Network or API issue. {e}"

def parse_address(address):
    """Parse the given address using NLP and detect potential PIN code."""
    entities = address_parser(address)
    
    parsed = {}
    for entity in entities:
        entity_type = entity['entity']  # Use 'entity' field
        entity_text = entity['word']  # Use 'word' field
        if entity_type not in parsed:
            parsed[entity_type] = []
        parsed[entity_type].append(entity_text)
    
    # Extract PIN code (6-digit number) from the address using regex
    pincode = None
    match = re.search(r'\b\d{6}\b', address)
    if match:
        pincode = match.group(0)
    
    return parsed, pincode

# -------------------------------
# Main OCR and Validation Loop
# -------------------------------

def main():
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
                logging.error(f"Error recognizing text: {e}")
                continue

            # Collect recognized text in a list
            text_paragraph = []

            # Draw bounding boxes and collect text
            for detection in result:
                top_left = tuple(map(int, detection[0][0]))
                bottom_right = tuple(map(int, detection[0][2]))
                text = detection[1]
                text_paragraph.append(text)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

            paragraph_output = " ".join(text_paragraph)
            translated_text = translator.translate(paragraph_output)
            parsed_components, pincode = parse_address(translated_text)
            
            if pincode:
                validation_result = validate_pincode(pincode, translated_text)
                print("\nValidation Result:")
                print(validation_result)

            cv2.imshow('Captured Text', frame)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
