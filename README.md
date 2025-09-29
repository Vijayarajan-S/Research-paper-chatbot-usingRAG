Research Paper Chatbot Using RAG

This project implements a research paper chatbot that answers questions based on the content of uploaded research papers. It uses Flask as the backend, HTML/CSS for the frontend, PyMuPDF for PDF processing, Pinecone for vector storage, and RAG models from Hugging Face with LangChain for intelligent responses.

Features

PDF Upload & Processing: Upload research papers in PDF format. Text is extracted using PyMuPDF.

Text Splitting: Documents are split into chunks using LangChain’s text splitter for efficient retrieval.

Vector Store: Stores embeddings in Pinecone for fast semantic search.

RAG Model: Uses Hugging Face language models with LangChain for context-aware answer generation.

Frontend Interface: Simple and responsive HTML/CSS interface to interact with the chatbot.

Flask API Endpoints: Handles PDF uploads, user queries, and returns answers from the chatbot.

Technologies Used

Python – Backend logic

Flask – API server

HTML/CSS/JS – Frontend interface

PyMuPDF – PDF text extraction

LangChain – Text splitting and RAG pipeline

Hugging Face – Pre-trained language models

Pinecone – Vector database for semantic search

Installation




Install dependencies:

pip install -r requirements.txt


Set up environment variables:

PINECONE_API_KEY – Your Pinecone API key

PINECONE_ENVIRONMENT – Pinecone environment (e.g., us-west1-gcp)

HUGGINGFACE_API_TOKEN – Hugging Face API token

Store these in a .env file in the root directory.

Usage

Run the Flask backend:

python app.py


Open the frontend:

Navigate to templates/index.html (Flask automatically serves this file).

Upload PDF research papers.

Ask questions in the chat interface.

Responses are fetched from the backend RAG model.

API Endpoints

POST /upload – Upload PDF files

POST /query – Send user questions and receive chatbot answers

The frontend interacts with these endpoints using JavaScript fetch or AJAX calls.

Project Structure
├── app.py                  # Main Flask backend
├── templates/
│   └── index.html          # Frontend HTML
├── pdf_processor.py        # PDF extraction using PyMuPDF
├── text_splitter.py        # Splits text into chunks
├── rag_model.py            # RAG pipeline using LangChain + Hugging Face
├── pinecone_index.py       # Pinecone vector store integration
├── requirements.txt
└── README.md
