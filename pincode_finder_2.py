import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def main():
    pincode = input("Enter the pincode: ").strip()  
    endpoint = f"https://api.postalpincode.in/pincode/{pincode}" 

    try:
        response = requests.get(endpoint)

        logging.info("API request sent, awaiting response.")

        if response.status_code != 200:
            logging.error(f"Unable to fetch data. HTTP Status Code: {response.status_code}")
            print(f"Error: Unable to fetch data. HTTP Status Code: {response.status_code}")
            return

        response_status = preprocess_response(response.text)
        logging.info(f"Response status after preprocessing: {response_status}")

        if response_status == "EmptyResponse":
            print("Error: Empty response from the server.")
        elif response_status == "ParseError":
            print("Error: Failed to parse JSON response.")
        elif response_status == "InvalidStructure":
            print("Error: Invalid response structure or no data found for this PIN code.")
        else:
            try:
                pincode_information = json.loads(response.text)
                necessary_information = pincode_information[0]["PostOffice"][0]

                logging.info("Successfully parsed response. Formatting output.")
                print("\nAPI Output (ML Processed):")
                for key, value in necessary_information.items():
                    print(f"{key}: {value}")
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON response. {e}")
    except requests.RequestException as e:
        logging.error(f"Network or API issue. {e}")
        print(f"Error: Network or API issue. {e}")

if __name__ == "__main__":
    main()
