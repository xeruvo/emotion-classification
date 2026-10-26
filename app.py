"""Optional local Gradio demo for a previously trained emotion model."""

from pathlib import Path

import gradio as gr
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

MODEL_DIR = Path("emotion_model")
LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise",
]


def load_model():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            "No trained model found. Run the notebook training section first "
            "and save the model to emotion_model/."
        )
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text: str):
    if not text or not text.strip():
        return "Enter text to classify."
    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(**inputs).logits)[0]
    ranked = sorted(zip(LABELS, probabilities.tolist()), key=lambda pair: pair[1], reverse=True)
    return "\n".join(f"{label}: {probability:.4f}" for label, probability in ranked[:5])


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, label="Text"),
    outputs=gr.Textbox(label="Top predicted emotions"),
    title="GoEmotions BERT classifier",
)

if __name__ == "__main__":
    demo.launch()
