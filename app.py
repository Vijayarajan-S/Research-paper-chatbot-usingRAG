from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import requests

# -------------------- LOAD ENV --------------------
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face API Token
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "research-bot"
EMBED_DIM = 384  # all-MiniLM-L6-v2 embedding dimension

# -------------------- INIT PINECONE --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# -------------------- INIT FLASK --------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------- INIT EMBEDDING MODEL --------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- FLASK ROUTES --------------------
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    if not text.strip():
        return jsonify({"status": "error", "message": "PDF is empty"}), 400

    chunks = split_text_into_chunks(text)
    embed_and_store_chunks(chunks)

    return jsonify({
        "status": "success",
        "message": f"{file.filename} uploaded and indexed.",
        "num_chunks": len(chunks),
        "sample_chunk": chunks[0]
    })


@app.route('/chat', methods=["POST"])
def chat():
    data = request.get_json()
    if 'question' not in data:
        return jsonify({"status": "error", "message": "No question provided"}), 400

    question = data['question']
    top_chunks = search_chunks(question, top_k=3)
    answer = generate_answer_qwen(question, top_chunks)

    return jsonify({
        "status": "success",
        "question": question,
        "answer": answer,
        "retrieved_chunks": top_chunks
    })

# -------------------- HELPER FUNCTIONS --------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def embed_and_store_chunks(chunks):
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        index.upsert(vectors=[(f"doc-{i}", embedding, {"text": chunk})])


def search_chunks(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]


def generate_answer_qwen(question, chunks):
    """
    Send retrieved chunks + user question to Hugging Face Qwen model.
    """
    context = " ".join(chunks)
    prompt = f"""
You are a helpful research assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}
Answer:
"""
    url = "https://api-inference.huggingface.co/models/Qwen/Qwen3-Next-80B-A3B-Instruct"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        response_json = response.json()
        if isinstance(response_json, list) and len(response_json) > 0 and 'generated_text' in response_json[0]:
            return response_json[0]['generated_text'].strip()
        else:
            print("HF response unexpected format:", response_json)
            return "Sorry, could not generate a response."

    except requests.exceptions.RequestException as e:
        print("HF request error:", str(e))
        return f"Error connecting to Hugging Face API: {str(e)}"


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(debug=True)
