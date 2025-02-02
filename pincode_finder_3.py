import requests
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# ------------------------------
# Step 1: Create or Load Dataset
# ------------------------------

# Simulate a dataset of PIN codes and their response status
data = {
    "pincode": ["560001", "110001", "123456", "000000"],
    "status": ["ValidResponse", "ValidResponse", "InvalidStructure", "EmptyResponse"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature extraction using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["pincode"])

# Encode target labels
y = df["status"].astype("category").cat.codes  # Encode labels as integers

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# Step 2: Train ML Model
# ------------------------------

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model and vectorizer
with open("ml_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Evaluate model accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ------------------------------
# Step 3: Use the ML Model in the Pipeline
# ------------------------------

# Load the model and vectorizer
with open("ml_model.pkl", "rb") as f:
    ml_model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def preprocess_pincode(pincode):
    """Preprocess the input PIN code."""
    return pincode.strip()

def classify_pincode(pincode):
    """Use the ML model to classify the PIN code."""
    vectorized_input = vectorizer.transform([pincode])
    prediction = ml_model.predict(vectorized_input)[0]
    status_mapping = dict(enumerate(df["status"].astype("category").cat.categories))
    return status_mapping[prediction]

# ------------------------------
# Step 4: Integration with API
# ------------------------------

# API Endpoint
ENDPOINT = "https://api.postalpincode.in/pincode/"

# Input PIN code
pincode = input("Enter your PIN code: ")

try:
    # Preprocessing step
    processed_pincode = preprocess_pincode(pincode)

    # ML model classification
    classification = classify_pincode(processed_pincode)

    # Fetch data if classification is valid
    if classification == "ValidResponse":
        response = requests.get(ENDPOINT + processed_pincode, timeout=10)
        if response.status_code != 200:
            print(f"Error: Unable to fetch data. HTTP Status Code: {response.status_code}")
        elif not response.text.strip():
            print("Error: Empty response from the server.")
        else:
            try:
                # Parse the JSON response
                pincode_information = json.loads(response.text)
                if not pincode_information or "PostOffice" not in pincode_information[0] or not pincode_information[0]["PostOffice"]:
                    print("Error: Invalid response structure or no data found for this PIN code.")
                else:
                    necessary_information = pincode_information[0]["PostOffice"][0]
                    print("\nAPI Output:")
                    for key, value in necessary_information.items():
                        print(f"{key}: {value}")
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON response. {e}")
    else:
        print(f"Error: {classification}")
except requests.RequestException as e:
    print(f"Error: Network or API issue. {e}")