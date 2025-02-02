import requests
import json

# API Key Interaction
ENDPOINT = "https://api.postalpincode.in/pincode/"
pincode = input("Enter your pincode: ")

try:
    response = requests.get(ENDPOINT + pincode)
    
    # Check if the response is successful
    if response.status_code != 200:
        print(f"Error: Unable to fetch data. HTTP Status Code: {response.status_code}")
    else:
        # Check if the response content is empty
        if not response.text.strip():
            print("Error: Empty response from the server.")
        else:
            # Attempt to parse JSON response
            try:
                pincode_information = json.loads(response.text)
                
                # Handle invalid response structure
                if not pincode_information or "PostOffice" not in pincode_information[0] or not pincode_information[0]["PostOffice"]:
                    print("Error: Invalid response structure or no data found for this PIN code.")
                else:
                    necessary_information = pincode_information[0]["PostOffice"][0]
                    print("\nAPI Output:")
                    for key, value in necessary_information.items():
                        print(f"{key}: {value}")
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON response. {e}")
except requests.RequestException as e:
    print(f"Error: Network or API issue. {e}")
