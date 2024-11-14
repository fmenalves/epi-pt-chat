import medspacy
from spacy.tokens import Span
from rules import brand, substance
from medspacy.ner import TargetRule, TargetMatcher
import spacy

# import en_ner_bc5cdr_md
import en_core_med7_lg


def setup_medspacy():
    """Set up the medSpaCy pipeline"""
    nlp = medspacy.load()
    Span.set_extension("code", default=None, force=True)
    Span.set_extension("system", default="SNOMED_CT", force=True)
    target_matcher = TargetMatcher(nlp)
    target_matcher.add(brand.brand_rules)

    # nlp.get_pipe("medspacy_target_matcher").add(brand.brand_rules)
    # nlp.get_pipe("medspacy_target_matcher").add(substance.substance_rules)

    return nlp, target_matcher


def preprocess(text: str):
    """Preprocess the HTML DOM"""
    nlp, target_matcher = setup_medspacy()

    doc = nlp(text)
    matches = target_matcher(doc)

    for match in matches:
        print(match)
        print(f"Text found: {match.text}, Category: {match.category}")

    if doc.ents:
        print(doc.ents)
        for ent in doc.ents:
            print(ent)

    return text


# print(preprocess("triticu ac pode ser partido?"))

nlp = spacy.load("en_core_med7_lg")


# This function generate anotation for each entities and label
def generate_annotation(texts):
    annotations = []
    for text in texts:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append((ent.start_char, ent.end_char, ent.label_))
        annotations.append((text, {"entities": entities}))
    return annotations


# Extract text entities and labels from the dataset (transcription)
medical_doc = [
    "triticu ac pode ser partido?",
    "AMlodipina é para tratar hiv e tosse",
    "trazodona serve para q?",
    "Brufen is for using data",
    "amlodipina serve para que?",
    "tosse é para tratar com paracetamol?",
]

# Let's generate annotations
annotations = generate_annotation(medical_doc)

# Let's print documents and annotations
print("Document:")
print(annotations)

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(
    "token-classification",
    model="Clinical-AI-Apollo/Medical-NER",
    aggregation_strategy="simple",
)
result = pipe(medical_doc)

print(result)


pipe = pipeline(
    "token-classification",
    model="pucpr/clinicalnerpt-medical",
    aggregation_strategy="simple",
)  #  Copy  # Load model directly
result = pipe(medical_doc)

print(result)

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(
    "token-classification",
    model="pineiden/nominal-groups-recognition-medical-disease-competencia2-bert-medical-ner",
    aggregation_strategy="simple",
)
result = pipe(medical_doc)

print(result)
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(
    "token-classification",
    model="m-daniyal-shaiq123/medical_prescription_lm",
    aggregation_strategy="simple",
)
result = pipe(medical_doc)

print(result)
