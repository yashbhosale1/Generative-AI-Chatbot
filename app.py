# app.py
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import openai
from utils import IntentClassifier, get_response_for_tag, clean_text
import traceback

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Optional: LangChain usage (simple example)
try:
    from langchain import OpenAI as LC_OpenAI
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

app = Flask(__name__)
intent_clf = IntentClassifier(intents_path="intents.json")

# a small system prompt to guide generation
SYSTEM_PROMPT = """You are a helpful assistant. Keep responses concise and friendly."""

def call_openai_chat(message: str, system_prompt: str = SYSTEM_PROMPT):
    """
    Calls OpenAI ChatCompletion (gpt-3.5/4 style)
    """
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set")
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.2,
            max_tokens=400
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json() or {}
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"ok": False, "error": "no message provided"}), 400

    # 1) Preprocess
    cleaned = clean_text(user_message)

    # 2) Intent detection
    try:
        tag, conf = intent_clf.predict(user_message)
    except Exception:
        tag, conf = "unknown", 0.0

    # If intent has a canned response and confidence high, return it
    if conf > 0.75:
        canned = get_response_for_tag(tag, intent_clf.tag_responses)
        if canned:
            return jsonify({
                "ok": True,
                "source": "intent_canned",
                "intent": tag,
                "confidence": conf,
                "reply": canned
            })

    # 3) Generate using LangChain (if available) else OpenAI directly
    try:
        if LANGCHAIN_AVAILABLE:
            # Example: use LangChain wrapper around OpenAI
            prompt = PromptTemplate(input_variables=["input"], template="{input}")
            llm = LC_OpenAI(temperature=0.2)
            chain = LLMChain(llm=llm, prompt=prompt)
            reply = chain.run(user_message)
            source = "langchain_openai"
        else:
            reply = call_openai_chat(user_message)
            source = "openai_chat"
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({
        "ok": True,
        "source": source,
        "intent": tag,
        "confidence": conf,
        "reply": reply
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
