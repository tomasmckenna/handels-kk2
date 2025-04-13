import os
import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
import subprocess
import tempfile
import time
import pickle
import hashlib
import json
import re
import curses

from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
BASE_FOLDER = os.getcwd()  # Search in the current directory
MODEL_PATH = "/home/tomas/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
LLAMA_CMD = "/home/tomas/llama.cpp/build/bin/llama-cli"

# Function to detect numbered folders
def find_numbered_folders():
    """Find folders that start with a number."""
    return [f for f in os.listdir(BASE_FOLDER) if os.path.isdir(f) and re.match(r'\d+', f)]

# Function to calculate file hash
def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

# Check if PDFs have changed
def have_pdfs_changed(pdf_folder, hash_file):
    if not os.path.exists(hash_file):
        return True
    
    with open(hash_file, 'r') as f:
        stored_hashes = json.load(f)
    
    current_files = {f: get_file_hash(os.path.join(pdf_folder, f)) for f in os.listdir(pdf_folder) if f.endswith(".pdf")}
    
    return stored_hashes != current_files

# Save new PDF hashes
def save_pdf_hashes(pdf_folder, hash_file):
    hashes = {f: get_file_hash(os.path.join(pdf_folder, f)) for f in os.listdir(pdf_folder) if f.endswith(".pdf")}
    with open(hash_file, 'w') as f:
        json.dump(hashes, f)

# Load PDFs and extract text
def load_pdfs(pdf_folder, vector_db_folder):
    index_file = os.path.join(vector_db_folder, "faiss_index.bin")
    vectorizer_file = os.path.join(vector_db_folder, "vectorizer.pkl")
    documents_file = os.path.join(vector_db_folder, "documents.pkl")
    filenames_file = os.path.join(vector_db_folder, "filenames.pkl")
    hash_file = os.path.join(vector_db_folder, "pdf_hashes.json")

    if all(os.path.exists(f) for f in [index_file, vectorizer_file, documents_file, filenames_file]) and not have_pdfs_changed(pdf_folder, hash_file):
        with open(documents_file, 'rb') as f:
            documents = pickle.load(f)
        with open(filenames_file, 'rb') as f:
            filenames = pickle.load(f)
        return documents, filenames
    
    documents, filenames = [], []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(pdf_folder, filename)) as doc:
                text = "\n".join([page.get_text("text") for page in doc])
                documents.append(text)
                filenames.append(filename)
    
    os.makedirs(vector_db_folder, exist_ok=True)
    with open(documents_file, 'wb') as f:
        pickle.dump(documents, f)
    with open(filenames_file, 'wb') as f:
        pickle.dump(filenames, f)
    save_pdf_hashes(pdf_folder, hash_file)
    
    return documents, filenames

# Create FAISS index
def create_search_index(documents, vector_db_folder):
    vectorizer_file = os.path.join(vector_db_folder, "vectorizer.pkl")
    index_file = os.path.join(vector_db_folder, "faiss_index.bin")

    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(documents).toarray()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))

    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    faiss.write_index(index, index_file)
    
    return vectorizer, index

# Retrieve documents
def retrieve_documents(query, vectorizer, index, documents, k=3):
    query_embedding = vectorizer.transform([query]).toarray().astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    return [documents[idx] for idx in indices[0] if idx < len(documents)]

# Generate response
def generate_with_llama(context, query):
    prompt = f"""
Context:
{context}

Question: {query}

Answer:
""".strip()
    
    command = [LLAMA_CMD, "-m", MODEL_PATH, "-p", prompt, "-n", "1024"]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return process.stdout.strip() if process.returncode == 0 else "Error running model"

# Run RAG pipeline
def rag_query(query, pdf_folder):
    vector_db_folder = os.path.join(pdf_folder, "vector_db")
    documents, filenames = load_pdfs(pdf_folder, vector_db_folder)
    vectorizer, index = create_search_index(documents, vector_db_folder)
    retrieved_docs = retrieve_documents(query, vectorizer, index, documents)
    context = "\n\n---\n\n".join(retrieved_docs)[:5000]
    return generate_with_llama(context, query) if context else "No relevant documents found."

# Menu UI with curses
def menu(stdscr):
    curses.curs_set(0)
    stdscr.clear()
    folders = find_numbered_folders()
    selected = 0

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Select a knowledge base:")
        for i, folder in enumerate(folders):
            mode = curses.A_REVERSE if i == selected else curses.A_NORMAL
            stdscr.addstr(i + 1, 2, folder, mode)
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and selected > 0:
            selected -= 1
        elif key == curses.KEY_DOWN and selected < len(folders) - 1:
            selected += 1
        elif key in [10, 13]:  # Enter key
            return folders[selected]

# Start tmux session
def start_tmux():
    session_name = "rag_tmux"
    os.system(f"tmux new-session -d -s {session_name} 'python {__file__}'")
    print(f"Started tmux session '{session_name}'. Attach with: tmux attach -t {session_name}")

if __name__ == "__main__":
    selected_folder = curses.wrapper(menu)
    query = input("Enter your query: ")
    print("\nProcessing query...")
    response = rag_query(query, os.path.join(BASE_FOLDER, selected_folder))
    print("\n=== ANSWER ===\n", response)
