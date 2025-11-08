# Generative AI Chatbot (Flask + OpenAI + LangChain + HuggingFace + scikit-learn)

## Overview
This project is a simple generative AI chatbot using:
- Flask: web server & endpoints
- OpenAI API: response generation (gpt-3.5-turbo or similar)
- LangChain: optional wrapper for advanced chaining
- HuggingFace / sentence-transformers: optional for embeddings or custom models
- scikit-learn: small intent classifier (Logistic Regression + TF-IDF)
- nltk: tokenization & preprocessing

## Setup
1. Clone/copy files.
2. Create and activate a Python virtualenv.
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
5. Run:
   ```bash
   python app.py
   ```
6. Open `http://localhost:5000/chat`.

## Notes & Extensions
- The `intents.json` dataset is small — expand patterns/responses for better intent coverage.
- To use LangChain functionality, install `langchain` and ensure compatible versions; the code automatically uses LangChain if import succeeds.
- If you want to use a HuggingFace local model for generation/embeddings, plug it where `call_openai_chat` is used (the code includes the `LANGCHAIN_AVAILABLE` branch).
- For production, switch debug off and consider security (rate limits, request validation, API key storage).
- Add streaming responses (server-sent events or WebSockets) for a more interactive UX.

## Files
- `app.py` — Flask app & routing
- `utils.py` — preprocessing & intent classifier
- `intents.json` — sample intents
- `templates/` — frontend templates
