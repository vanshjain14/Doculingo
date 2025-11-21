# app.py
import os
import sqlite3
import multiprocessing
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import re
# transformers translation
from transformers import MarianTokenizer, MarianMTModel

# langchain helpers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------- BASIC SETUP --------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
VECTOR_STORE_PATH = "vector_store"
DB_PATH = "feedback.db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------- MODEL PATHS --------------------
file_dir = os.path.dirname(os.path.abspath(__file__))

LLM_MODEL_PATH = os.path.join(file_dir, "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


EN_ES = "Helsinki-NLP/opus-mt-en-es"
ES_EN = "Helsinki-NLP/opus-mt-es-en"
EN_DE = "Helsinki-NLP/opus-mt-en-de"
DE_EN = "Helsinki-NLP/opus-mt-de-en"

# -------------------- DATABASE --------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                question TEXT,
                answer TEXT,
                rating TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_embeddings (
                id INTEGER PRIMARY KEY,
                question TEXT,
                answer TEXT,
                embedding BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()

# -------------------- LOAD MODELS --------------------
print("Loading translation models (may attempt to load from HF cache)...")
# tokenizers + small translation models
en_es_tok = MarianTokenizer.from_pretrained(EN_ES)
en_es_model = MarianMTModel.from_pretrained(EN_ES).to(DEVICE)

es_en_tok = MarianTokenizer.from_pretrained(ES_EN)
es_en_model = MarianMTModel.from_pretrained(ES_EN).to(DEVICE)

en_de_tok = MarianTokenizer.from_pretrained(EN_DE)
en_de_model = MarianMTModel.from_pretrained(EN_DE).to(DEVICE)

de_en_tok = MarianTokenizer.from_pretrained(DE_EN)
de_en_model = MarianMTModel.from_pretrained(DE_EN).to(DEVICE)

print("Loading embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vector_store = None
if os.path.exists(VECTOR_STORE_PATH):
    try:
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded.")
    except Exception as ex:
        print("⚠️ Could not load vector store:", ex)

# LlamaCpp settings balanced for CPU
cpu_count = os.cpu_count() or 4
n_threads = min(8, cpu_count)  
print(f"Loading LLM (threads={n_threads})...")
llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    n_ctx=2048,
    n_threads=n_threads,
    n_batch=64,
    f16_kv=True,
    temperature=0.2,
    max_tokens=400,  
)
print("✅ LLM ready.")

# -------------------- TRANSLATION HELPERS --------------------
def translate_to_en(text: str, lang: str) -> str:
    if not text:
        return ""
    if lang == "es":
        tokens = es_en_model.generate(**es_en_tok(text, return_tensors="pt", padding=True).to(DEVICE))
        return es_en_tok.decode(tokens[0], skip_special_tokens=True)
    if lang == "de":
        tokens = de_en_model.generate(**de_en_tok(text, return_tensors="pt", padding=True).to(DEVICE))
        return de_en_tok.decode(tokens[0], skip_special_tokens=True)
    return text

def translate_from_en(text: str, lang: str) -> str:
    if not text:
        return ""
    if lang == "es":
        tokens = en_es_model.generate(**en_es_tok(text, return_tensors="pt", padding=True).to(DEVICE))
        return en_es_tok.decode(tokens[0], skip_special_tokens=True)
    if lang == "de":
        tokens = en_de_model.generate(**en_de_tok(text, return_tensors="pt", padding=True).to(DEVICE))
        return en_de_tok.decode(tokens[0], skip_special_tokens=True)
    return text

# -------------------- UTILITIES --------------------
def strip_answer_prefix(s: str) -> str:
    if not s:
        return s
    s_strip = s.strip()
    lowered = s_strip.lower()
    # remove common prefixes like "answer:", "respuesta:", "response:"
    prefixes = ["answer:", "respuesta:", "response:", "antwort:", "respuesta", "antwort"]
    for p in prefixes:
        if lowered.startswith(p):
            return s_strip[len(p):].strip()
    return s_strip

def truncate_to_last_sentence(text: str, min_keep_chars: int = 40) -> str:
    """
    Return text truncated at last full sentence terminator (., ?, !).
    If none found and text is long, return at last whitespace within reasonable length.
    """
    if not text:
        return text
    # prefer splitting at ., ?, !
    last_pos = -1
    for ch in ('.', '?', '!'):
        pos = text.rfind(ch)
        if pos > last_pos:
            last_pos = pos
    if last_pos >= 0 and last_pos + 1 >= min_keep_chars:
        return text[: last_pos + 1].strip()
    # fallback: if text shorter than 300 chars return as-is
    if len(text) <= 300:
        return text.strip()
    # otherwise cut at last period-like punctuation before 300-500 chars
    for pos in range(min(len(text), 500) - 1, 100, -1):
        if text[pos] in '.!?':
            return text[:pos+1].strip()
    # final fallback: cut at nearest space to 300 chars
    idx = text.rfind(' ', 280, 350)
    if idx != -1:
        return text[:idx].strip()
    return text[:400].strip()

# -------------------- ROUTES --------------------
@app.route("/upload", methods=["POST"])
def upload_pdf():
    global vector_store
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        vector_store = FAISS.from_documents(chunks, embedding_model)
        vector_store.save_local(VECTOR_STORE_PATH)
        return jsonify({"message": "✅ File processed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    global vector_store
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    language = data.get("language", "en") or "en"

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if vector_store is None:
        return jsonify({"error": "Upload a document first"}), 400

    try:
        # Translate incoming question into English if UI language is not English
        q_en = question if language == "en" else translate_to_en(question, language)

        # Retrieve top chunks (k small to keep it fast)
        retrieved = vector_store.similarity_search(q_en, k=4)
        context = "\n\n".join([d.page_content[:600] for d in retrieved])

        # Clear and explicit prompt; instruct not to prefix with "Answer:"
        prompt = f"""
<s>[INST]
Use ONLY the context below to answer the question. Answer in 2-4 clear sentences.
If the answer isn't in the context, respond: "I don't have enough information."
Do NOT prefix the response with "Answer:", "Response:", or any labels.
Keep the reply factual and complete. Do NOT cut sentences midway.
[/INST]

Context:
{context}

Question:
{q_en}
"""

        # Ask LLM (allow more tokens for answers so they don't cut)
        answer_en = llm.invoke(prompt, max_tokens=400).strip()
        answer_en = strip_answer_prefix(answer_en)
        answer_en = truncate_to_last_sentence(answer_en)

        # Translate back to the selected UI language if needed
        answer = answer_en if language == "en" else translate_from_en(answer_en, language)

        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    global vector_store
    data = request.get_json()
    language = data.get("language", "en")

    if vector_store is None:
        return jsonify({"error": "Please upload a PDF first."}), 400

    try:
        # --------- Get chunks in original reading order ----------
        docs = list(vector_store.docstore._dict.values())
        docs_sorted = sorted(docs, key=lambda d: d.metadata.get("page", 0))

        # --------- Use first 2–3 pages only ----------
        selected = docs_sorted[:3]

        raw_context = "\n\n".join(doc.page_content for doc in selected)

        # --------- Trim safely without cutting sentences ----------
        def safe_trim(text, limit=1800):
            text = text[:limit]
            end = text.rfind(".")
            return text[: end + 1] if end > 0 else text

        context = safe_trim(raw_context, 1800)

        # --------- LLM prompt ----------
        prompt = f"""
<s>[INST]
Summarize the document into **3-4 complete sentences**.
Do not cut sentences.  
Keep total length under **120 words**.  
Only use the provided context.

Context:
{context}
[/INST]
"""

        summary_en = llm.invoke(prompt).strip()

        # --------- Manual language choice (no autodetect) ----------
        if language == "es":
            return jsonify({"summary": translate_from_en(summary_en, "es")})
        elif language == "de":
            return jsonify({"summary": translate_from_en(summary_en, "de")})
        else:
            return jsonify({"summary": summary_en})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------------------
# /suggest-questions  (Offline, 2 questions only)
# -----------------------------------------
@app.route("/suggest-questions", methods=["POST"])
def suggest_questions():
    global vector_store
    try:
        data = request.get_json()
        user_question = (data.get("question") or "").strip()
        language = data.get("language", "en")

        if vector_store is None:
            return jsonify({"suggestions": []})

        # Translate user question → English for vector search
        q_en = user_question if language == "en" else translate_to_en(user_question, language)

        # Retrieve context
        retrieved = vector_store.similarity_search(q_en, k=3)
        context = "\n\n".join([d.page_content[:500] for d in retrieved])

        # LLM prompt (STRICT)
        prompt = f"""
<s>[INST]
Your job is to generate exactly **2 short follow-up questions** based ONLY on the document context.

STRICT RULES:
- Output MUST be in the language: "{language}".
- NO English content if language ≠ English.
- NO parentheses, NO duplicated translations.
- NO numbering, NO bullets.
- Each question must be **5–10 words max**.
- MUST be relevant to both the context and the user's question.
- MUST NOT repeat the user's question.
- MUST NOT include answers.
- MUST output EXACTLY two lines. No extra text.

CONTEXT:
{context}

USER QUESTION:
{user_question}

Return ONLY the 2 questions, each on one line.
[/INST]
"""

        raw_output = llm.invoke(prompt).strip()

        # Split output into non-empty clean lines
        lines = [l.strip() for l in raw_output.split("\n") if l.strip()]

        cleaned = []
        for l in lines:
            # remove bullets, numbers, parentheses translations
            l = re.sub(r"^\s*[\-\•\*\d\.\)\(]+\s*", "", l)
            # remove duplicated translation inside parentheses
            l = re.sub(r"\([^)]*\)", "", l)
            l = l.strip()
            if l:
                cleaned.append(l)
            if len(cleaned) == 2:
                break

        # Ensure exactly 2
        while len(cleaned) < 2:
            cleaned.append("")

        # Translate if needed
        if language != "en":
            cleaned = [translate_from_en(q, language) if q else "" for q in cleaned]

        return jsonify({"suggestions": cleaned})

    except Exception as e:
        print("Suggestion error:", e)
        return jsonify({"suggestions": []})



@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json() or {}
    question = data.get("question")
    answer = data.get("answer")
    rating = data.get("rating")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO feedback (question, answer, rating) VALUES (?, ?, ?)", (question, answer, rating))
            conn.commit()
        return jsonify({"message": "✅ Feedback saved"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- MAIN --------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    app.run(host="0.0.0.0", port=5000, debug=False)
