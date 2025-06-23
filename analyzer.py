import spacy
nlp=spacy.load('en_core_web_sm')
def extract_entities(texts):
    result = []
    for text in texts:
        doc = nlp(text)
        entities = [(ent.text,ent.label_) for ent in doc.ents]
        result.append({"text":text,"entities":entities})
    return result