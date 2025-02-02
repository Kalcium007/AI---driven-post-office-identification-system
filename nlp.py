from transformers import pipeline

# Load the pre-trained model for token classification
nlp = pipeline("ner", model="bert-base-multilingual-cased", tokenizer="bert-base-multilingual-cased")

# Sample text
text = "Flat No 501, Mahal Ingapura, Coimbatore, Karnataka 560038"

# Run the Named Entity Recognition (NER) pipeline
entities = nlp(text)

# Print extracted entities
print("Extracted Entities:", entities)

# Filter out locations based on the entity labels
location_entities = []

# Iterate over each entity in the result
for entity in entities:
    # Check if the entity belongs to a location category
    # Since the model might not output 'B-LOC' or 'I-LOC', we use entity['entity'] which might be 'LABEL_X'
    if 'LOC' in entity['entity']:  # This checks if 'LOC' is part of the entity type (for location)
        location_entities.append(entity['word'])

# Print the extracted location entities
print("Location Entities:", location_entities)
