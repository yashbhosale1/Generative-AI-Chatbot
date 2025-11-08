# utils.py
import re
import json
import os
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Dict, Any
import random

nltk.download('punkt', quiet=True)

# --- Preprocessing functions ---
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return word_tokenize(text)

# --- Intent detection (train on small dataset on startup) ---
class IntentClassifier:
    def __init__(self, intents_path="intents.json"):
        self.intents_path = intents_path
        self.vectorizer = TfidfVectorizer()
        self.clf = LogisticRegression(max_iter=1000)
        self.tag_responses = {}
        self._train()

    def _train(self):
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"{self.intents_path} not found")
        data = json.load(open(self.intents_path, 'r', encoding='utf-8'))
        patterns = []
        tags = []
        for intent in data.get("intents", []):
            tag = intent["tag"]
            for p in intent["patterns"]:
                patterns.append(clean_text(p))
                tags.append(tag)
            # store example responses
            self.tag_responses[tag] = intent.get("responses", [])
        if not patterns:
            return
        X = self.vectorizer.fit_transform(patterns)
        self.clf.fit(X, tags)

    def predict(self, text: str) -> Tuple[str, float]:
        x = self.vectorizer.transform([clean_text(text)])
        proba = self.clf.predict_proba(x)[0]
        idx = proba.argmax()
        tag = self.clf.classes_[idx]
        return tag, float(proba[idx])

# --- Small helper to pick response from tag if applicable ---
def get_response_for_tag(tag: str, tag_responses: Dict[str, Any]) -> str:
    responses = tag_responses.get(tag, [])
    return random.choice(responses) if responses else ""
