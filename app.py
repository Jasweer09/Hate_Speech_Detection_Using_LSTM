from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import re
import numpy as np
import uvicorn

# ---------------- CONFIG ----------------
MODEL_PATH = "hate_speech_model.h5"
VOCAB_SIZE = 10000
MAX_LEN = 20
DIMENSION = 50  # not directly needed at inference
LABEL_MAP = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="Hate Speech Detection API", version="1.0")

class InputText(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float

# ---------------- PREPROCESSING ----------------
def clean_text(text: str) -> str:
    # Remove unwanted characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Lemmatization
    doc = nlp(text.lower())
    lemma_text = " ".join([token.lemma_ for token in doc])

    # Remove stopwords
    doc = nlp(lemma_text)
    final_text = " ".join([token.text for token in doc if not token.is_stop])

    return final_text

def text_to_sequence(text: str):
    cleaned = clean_text(text)
    one_hot_rep = [one_hot(cleaned, VOCAB_SIZE)]
    padded = pad_sequences(one_hot_rep, maxlen=MAX_LEN, padding="post")
    return padded

# ---------------- INFERENCE ----------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: InputText):
    try:
        seq = text_to_sequence(input_data.text)
        preds = model.predict(seq)
        confidence = float(np.max(preds))
        label = int(np.argmax(preds, axis=1)[0])

        return {
            "text": input_data.text,
            "prediction": LABEL_MAP.get(label, "Unknown"),
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
