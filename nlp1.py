from transformers import pipeline

# Load pretrained NLP model
address_parser = pipeline("ner", model="dslim/bert-base-NER")

# Example address
address = "123 MG Road, Bangalore, Karnataka, 560001"

# Extract entities
entities = address_parser(address)
print(f"Parsed Components: {entities}")