from flask import Flask, request, jsonify
import cv2
import numpy as np
from ocr_tamil.ocr import OCR
import base64

app = Flask(__name__)
ocr = OCR(detect=True)  # Initialize OCR model

@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data.get('image', '').split(',')[1]  # Extract base64 image data

    try:
        # Decode the image data from base64
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save the received image for debugging
        temp_image_path = 'received_image.png'
        cv2.imwrite(temp_image_path, img)
        print(f"Image saved at: {temp_image_path}")

        # Perform OCR
        extracted_texts = ocr.predict(temp_image_path)
        print(f"OCR Output: {extracted_texts}")

        return jsonify({"text": " ".join(extracted_texts) if extracted_texts else "No text detected"})
    except Exception as e:
        print(f"Error during image processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
