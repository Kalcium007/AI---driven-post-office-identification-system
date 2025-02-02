from transformers import pipeline

# Load the IndicNER model for NER task
nlp = pipeline("ner", model="ai4bharat/IndicNER", tokenizer="ai4bharat/IndicNER")

# Sample Tamil text
text = "சென்னையில் பள்ளிகளும் கல்லூரிகளும் திறக்கப்படுகின்றன."

# Run NER on the text
ner_results = nlp(text)

# Print the results
print("Named Entities:", ner_results)
