import os
import sys
import pymupdf as fitz  # PyMuPDF
import faiss
import numpy as np
import subprocess
import pickle
import hashlib
import json
import curses
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import io
import re
import tempfile

# Constants
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRECTORY = "/home/tomas/models/mistral/"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
LLAMA_CMD = "/home/tomas/llama.cpp/build/bin/llama-cli"

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'txt': 'text/plain',
    'html': 'text/html',
    'htm': 'text/html',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'xls': 'application/vnd.ms-excel',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'csv': 'text/csv',
}

# Setup logging
def setup_logging():
    log_dir = os.path.join(BASE_FOLDER, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"rag_system_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

LOG_FILE = setup_logging()

def log_system_info():
    logging.info(f"Python version: {sys.version}")
    logging.info(f"System platform: {sys.platform}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Base folder: {BASE_FOLDER}")
    logging.info(f"Files in base folder: {os.listdir(BASE_FOLDER)}")

log_system_info()

def find_knowledge_base_folders():
    """Find all folders that contain supported document files, excluding vector_db folders"""
    folders = []
    logging.info(f"Searching for knowledge bases in: {BASE_FOLDER}")
    
    for item in os.listdir(BASE_FOLDER):
        full_path = os.path.join(BASE_FOLDER, item)
        
        if os.path.isdir(full_path) and item != 'vector_db' and item != 'logs':
            # Find all documents recursively
            document_files = find_documents_recursive(full_path)
            
            if document_files:
                logging.info(f"Found knowledge base: {item}")
                logging.info(f"Document files: {len(document_files)} files found")
                folders.append(item)
    
    if not folders:
        logging.warning("No knowledge base folders found!")
        logging.warning(f"Current directory contents: {os.listdir(BASE_FOLDER)}")
    
    return folders

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def have_documents_changed(kb_folder, hash_file):
    if not os.path.exists(hash_file):
        return True
    
    with open(hash_file, 'r') as f:
        stored_hashes = json.load(f)
    
    # Get all document files recursively
    current_files = find_documents_recursive(kb_folder)
    
    # If the number of files has changed, assume the collection has changed
    if len(stored_hashes) != len(current_files):
        return True
    
    # Check if any file hashes have changed
    for filepath, _ in current_files:
        rel_path = os.path.relpath(filepath, kb_folder)
        file_hash = get_file_hash(filepath)
        
        if rel_path not in stored_hashes or stored_hashes[rel_path] != file_hash:
            return True
    
    return False

def save_document_hashes(kb_folder, hash_file):
    document_files = find_documents_recursive(kb_folder)
    hashes = {}
    
    for filepath, rel_path in document_files:
        hashes[rel_path] = get_file_hash(filepath)
    
    with open(hash_file, 'w') as f:
        json.dump(hashes, f)

def extract_text_from_pdf(filepath):
    """Extract text from PDF file using PyMuPDF"""
    try:
        with fitz.open(filepath) as doc:
            text = "\n".join([page.get_text("text") for page in doc])
            return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {filepath}: {e}")
        return ""

def extract_text_from_txt(filepath):
    """Extract text from plain text file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error extracting text from TXT {filepath}: {e}")
        return ""

def extract_text_from_html(filepath):
    """Extract text from HTML file - simple version"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
            # Simple HTML tag removal - for better results consider using BeautifulSoup
            text = re.sub(r'<[^>]+>', ' ', html_content)
            text = re.sub(r'\s+', ' ', text)
            return text
    except Exception as e:
        logging.error(f"Error extracting text from HTML {filepath}: {e}")
        return ""

def extract_text_from_csv(filepath):
    """Extract text from CSV file"""
    try:
        text_output = io.StringIO()
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                text_output.write(" | ".join(row) + "\n")
        return text_output.getvalue()
    except Exception as e:
        logging.error(f"Error extracting text from CSV {filepath}: {e}")
        return ""

def extract_text_from_office_document(filepath):
    """
    Extract text from Microsoft Office documents using PyMuPDF
    This requires convertng the document to PDF first
    For production use, consider using python-docx, openpyxl, etc.
    """
    try:
        # Check if textract library is available as an alternative
        try:
            import textract
            text = textract.process(filepath).decode('utf-8')
            return text
        except ImportError:
            logging.warning(f"textract not available, using fallback for {filepath}")
            
            # For simplicity, just extract what we can based on file extension
            ext = filepath.lower().split('.')[-1]
            if ext in ['doc', 'docx']:
                # Simple hack: read as binary and extract some text
                with open(filepath, 'rb') as f:
                    content = f.read()
                    text = content.decode('utf-8', errors='ignore')
                    # Clean up non-printable characters
                    text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
                    return text
            elif ext in ['xls', 'xlsx']:
                logging.warning(f"Proper Excel extraction requires openpyxl. Limited text extracted from {filepath}")
                return f"[Limited text extraction from {os.path.basename(filepath)}]"
                
    except Exception as e:
        logging.error(f"Error extracting text from {filepath}: {e}")
    
    return ""

def extract_text_from_file(filepath):
    """Extract text from a file based on its extension"""
    ext = filepath.lower().split('.')[-1]
    
    if ext == 'pdf':
        return extract_text_from_pdf(filepath)
    elif ext == 'txt':
        return extract_text_from_txt(filepath)
    elif ext in ['html', 'htm']:
        return extract_text_from_html(filepath)
    elif ext == 'csv':
        return extract_text_from_csv(filepath)
    elif ext in ['doc', 'docx', 'xls', 'xlsx']:
        return extract_text_from_office_document(filepath)
    else:
        logging.warning(f"Unsupported file type: {ext} - {filepath}")
        return ""

def find_documents_recursive(folder_path):
    """Find all supported document files recursively within a folder"""
    documents = []
    logging.info(f"Starting deep scan in: {folder_path}")

    for root, dirs, files in os.walk(folder_path):
        logging.info(f"Scanning folder: {root}")
        
        # Skip 'vector_db' folders
        if os.path.basename(root) == "vector_db":
            logging.info(f"Skipping vector_db folder: {root}")
            continue
        
        for file in files:
            ext = file.lower().split('.')[-1]
            if ext in SUPPORTED_EXTENSIONS:
                filepath = os.path.join(root, file)
                relative_path = os.path.relpath(filepath, folder_path)
                logging.info(f"Found file: {relative_path}")
                documents.append((filepath, relative_path))
            else:
                logging.debug(f"Ignoring unsupported file: {file}")

    logging.info(f"Total files discovered: {len(documents)}")
    return documents


def load_documents(kb_folder, vector_db_folder):
    """Load documents from the specified folder and its subfolders"""
    index_file = os.path.join(vector_db_folder, "faiss_index.bin")
    vectorizer_file = os.path.join(vector_db_folder, "vectorizer.pkl")
    documents_file = os.path.join(vector_db_folder, "documents.pkl")
    filenames_file = os.path.join(vector_db_folder, "filenames.pkl")
    hash_file = os.path.join(vector_db_folder, "document_hashes.json")

    os.makedirs(vector_db_folder, exist_ok=True)

    logging.info(f"\n📂 Loading knowledge base: {kb_folder}")
    logging.info(f"🔎 Vector DB path: {vector_db_folder}")
    logging.info("📁 Checking for existing vector data...")

    if all(os.path.exists(f) for f in [index_file, vectorizer_file, documents_file, filenames_file]) and not have_documents_changed(kb_folder, hash_file):
        logging.info("✅ Preprocessed data exists. Loading from cache.")
        with open(documents_file, 'rb') as f:
            documents = pickle.load(f)
        with open(filenames_file, 'rb') as f:
            filenames = pickle.load(f)
        return documents, filenames

    logging.info("🔄 Reprocessing documents due to change or missing data.")
    documents, filenames = [], []

    # Discover all document files (deep scan)
    document_files = find_documents_recursive(kb_folder)

    if not document_files:
        logging.warning("⚠️ No supported documents found during scan!")

    for filepath, rel_path in document_files:
        try:
            text = extract_text_from_file(filepath)
            if text:
                documents.append(text)
                filenames.append(rel_path)
                logging.info(f"📄 Document added: {rel_path}")
            else:
                logging.warning(f"⚠️ No text extracted from: {rel_path}")
        except Exception as e:
            logging.error(f"❌ Error processing {rel_path}: {e}")

    logging.info(f"\n✅ Finished processing. {len(documents)} documents added to the RAG system.")
    logging.info("📃 Final document list:")
    for doc_name in filenames:
        logging.info(f"  - {doc_name}")

    # Save data for future use
    with open(documents_file, 'wb') as f:
        pickle.dump(documents, f)
    with open(filenames_file, 'wb') as f:
        pickle.dump(filenames, f)
    save_document_hashes(kb_folder, hash_file)

    return documents, filenames

def create_search_index(documents, vector_db_folder):
    vectorizer_file = os.path.join(vector_db_folder, "vectorizer.pkl")
    index_file = os.path.join(vector_db_folder, "faiss_index.bin")

    if os.path.exists(vectorizer_file) and os.path.exists(index_file):
        logging.info("Loading existing search index...")
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        index = faiss.read_index(index_file)
        return vectorizer, index

    logging.info("Creating new search index...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)    
    
    embeddings = vectorizer.fit_transform(documents).toarray()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))

    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    faiss.write_index(index, index_file)
    
    return vectorizer, index

def retrieve_documents(query, vectorizer, index, documents, k=3):
    query_embedding = vectorizer.transform([query]).toarray().astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    return [documents[idx] for idx in indices[0] if idx < len(documents)]

def find_model_files():
    """Find all .gguf model files in the model directory."""
    return [f for f in os.listdir(MODEL_DIRECTORY) if f.endswith('.gguf')]

class RAGInterface:
    def __init__(self):
        self.current_model = DEFAULT_MODEL_PATH
        self.current_prompt_template = """Answer the question based on the following context:

Context:
{context}

Question: {query}

Answer the question with bullet points and always be comprehensive and cite the source documents:"""
        self.selected_kb = None
        logging.info("RAGInterface initialized")

    def safe_addstr(self, window, y, x, text, attr=0):
        """Safely add string to window with bounds checking"""
        try:
            h, w = window.getmaxyx()
            if y >= 0 and y < h and x >= 0 and x < w:
                text = text[:w-x-1]
                window.addstr(y, x, text, attr)
                return True
        except curses.error:
            pass
        return False

    def display_main_menu(self, stdscr):
        """Main menu with multiple options"""
        curses.curs_set(0)
        selected = 0
        menu_options = [
            "Select Knowledge Base",
            "Change Model",
            "Edit Prompt Template",
            "Run Query",
            "Exit"
        ]

        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            
            # Check minimum terminal size
            if h < 12 or w < 40:
                self.display_error(stdscr, "Terminal too small! (Min 40x12)")
                stdscr.getch()
                continue
                
            # Display title
            title = "RAG QUERY INTERFACE"
            self.safe_addstr(stdscr, 1, (w - len(title)) // 2, title, curses.A_BOLD | curses.color_pair(2))
            
            # Display current selections
            info_lines = [
                f"Current Model: {os.path.basename(self.current_model)}",
                f"Knowledge Base: {os.path.basename(self.selected_kb) if self.selected_kb else 'None'}"
            ]
            for i, info in enumerate(info_lines):
                self.safe_addstr(stdscr, 3 + i, 2, info, curses.color_pair(3))

            # Display menu options
            for i, option in enumerate(menu_options):
                y_pos = 7 + i
                if y_pos >= h:
                    continue
                mode = curses.A_REVERSE if i == selected else curses.A_NORMAL
                x_pos = max(0, (w - len(option)) // 2)
                self.safe_addstr(stdscr, y_pos, x_pos, option, mode | curses.color_pair(1))

            stdscr.refresh()

            key = stdscr.getch()
            if key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(menu_options) - 1:
                selected += 1
            elif key in [10, 13]:  # Enter key
                if menu_options[selected] == "Select Knowledge Base":
                    self.selected_kb = self.select_knowledge_base(stdscr)
                elif menu_options[selected] == "Change Model":
                    self.change_model(stdscr)
                elif menu_options[selected] == "Edit Prompt Template":
                    self.edit_prompt_template(stdscr)
                elif menu_options[selected] == "Run Query":
                    self.run_query(stdscr)
                elif menu_options[selected] == "Exit":
                    return None
            elif key == 27:  # Escape key
                return None

    def select_knowledge_base(self, stdscr):
        """Select a knowledge base folder"""
        folders = find_knowledge_base_folders()
        
        if not folders:
            self.display_error(stdscr, "No knowledge base folders found!")
            return None

        selected = 0
        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            
            self.safe_addstr(stdscr, 1, (w - len("Select Knowledge Base")) // 2, 
                           "Select Knowledge Base", curses.A_BOLD | curses.color_pair(2))

            for i, folder in enumerate(folders):
                y_pos = 3 + i
                if y_pos >= h:
                    continue
                mode = curses.A_REVERSE if i == selected else curses.A_NORMAL
                x_pos = max(0, (w - len(folder)) // 2)
                self.safe_addstr(stdscr, y_pos, x_pos, folder, mode | curses.color_pair(1))

            self.safe_addstr(stdscr, h-2, 2, 
                           "↑/↓: Navigate | Enter: Select | Esc: Cancel", curses.A_DIM)
            stdscr.refresh()

            key = stdscr.getch()
            if key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(folders) - 1:
                selected += 1
            elif key in [10, 13]:  # Enter key
                return os.path.join(BASE_FOLDER, folders[selected])
            elif key == 27:  # Escape key
                return None

    def change_model(self, stdscr):
        """Change the current model"""
        models = find_model_files()
        if not models:
            self.display_error(stdscr, "No model files found!")
            return

        selected = 0
        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            
            self.safe_addstr(stdscr, 1, (w - len("Select Model")) // 2, 
                           "Select Model", curses.A_BOLD | curses.color_pair(2))

            for i, model in enumerate(models):
                y_pos = 3 + i
                if y_pos >= h:
                    continue
                mode = curses.A_REVERSE if i == selected else curses.A_NORMAL
                x_pos = max(0, (w - len(model)) // 2)
                self.safe_addstr(stdscr, y_pos, x_pos, model, mode | curses.color_pair(1))

            self.safe_addstr(stdscr, h-2, 2, 
                           "↑/↓: Navigate | Enter: Select | Esc: Cancel", curses.A_DIM)
            stdscr.refresh()

            key = stdscr.getch()
            if key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(models) - 1:
                selected += 1
            elif key in [10, 13]:  # Enter key
                self.current_model = os.path.join(MODEL_DIRECTORY, models[selected])
                return
            elif key == 27:  # Escape key
                return

    def edit_prompt_template(self, stdscr):
        """Safely edit the prompt template"""
        curses.curs_set(1)
        curses.echo()
        h, w = stdscr.getmaxyx()
        
        # Clear screen and display instructions
        stdscr.clear()
        self.safe_addstr(stdscr, 1, 2, "Edit Prompt Template (must include {context} and {query}):")
        self.safe_addstr(stdscr, 3, 2, "Current Template:")
        self.safe_addstr(stdscr, 4, 2, self.current_prompt_template)
        
        # Create input window with safe dimensions
        input_height = min(5, h-6)
        input_width = min(w-4, 80)
        input_win = curses.newwin(input_height, input_width, 6, 2)
        input_win.keypad(1)
        
        # Initialize with current template
        template = self.current_prompt_template
        try:
            input_win.addstr(0, 0, template[:input_width*input_height-1])
        except curses.error:
            pass
        
        input_win.refresh()
        
        # Edit loop
        pos = len(template)
        while True:
            try:
                input_win.move(0, min(pos, input_width-1))
                ch = input_win.getch()
                
                if ch == 7:  # Ctrl+G - Save
                    if "{context}" in template and "{query}" in template:
                        self.current_prompt_template = template
                        break
                    else:
                        self.display_error(stdscr, "Template must include {context} and {query}!")
                        continue
                elif ch == 27:  # Escape - Cancel
                    break
                elif ch == curses.KEY_BACKSPACE or ch == 127:
                    if pos > 0:
                        template = template[:pos-1] + template[pos:]
                        pos -= 1
                elif ch == curses.KEY_LEFT:
                    pos = max(0, pos-1)
                elif ch == curses.KEY_RIGHT:
                    pos = min(len(template), pos+1)
                elif ch >= 32 and ch <= 126:  # Printable characters
                    template = template[:pos] + chr(ch) + template[pos:]
                    pos += 1
                
                # Redraw template
                input_win.clear()
                try:
                    input_win.addstr(0, 0, template[:input_width*input_height-1])
                except curses.error:
                    pass
                input_win.refresh()
                
            except curses.error as e:
                logging.error(f"Input error: {str(e)}")
                continue
        
        curses.noecho()
        curses.curs_set(0)

    def run_query(self, stdscr):
        """Run a RAG query and display the informed answer"""
        try:
            curses.echo()
            h, w = stdscr.getmaxyx()
            
            self.safe_addstr(stdscr, 1, 2, "Enter Query:")
            query = ""
            while True:
                try:
                    stdscr.move(3, 2)
                    stdscr.clrtoeol()
                    query = stdscr.getstr(3, 2, w-4).decode('utf-8')
                    break
                except curses.error:
                    continue
            
            if not query:
                return
            
            if not self.selected_kb:
                self.selected_kb = self.select_knowledge_base(stdscr)
                if not self.selected_kb:
                    return
            
            stdscr.clear()
            self.safe_addstr(stdscr, 1, 2, "Processing query...")
            stdscr.refresh()
            
            try:
                vector_db_folder = os.path.join(self.selected_kb, "vector_db")
                documents, filenames = load_documents(self.selected_kb, vector_db_folder)
                
                if not documents:
                    self.display_error(stdscr, "No documents found in knowledge base!")
                    return
                    
                vectorizer, index = create_search_index(documents, vector_db_folder)
                retrieved_docs = retrieve_documents(query, vectorizer, index, documents)
                
                # Show which files were retrieved
                stdscr.clear()
                self.safe_addstr(stdscr, 1, 2, "Relevant documents found:")
                doc_indices = [documents.index(doc) for doc in retrieved_docs if doc in documents]
                for i, idx in enumerate(doc_indices):
                    if i < 5:  # Limit to 5 documents to avoid screen overflow
                        self.safe_addstr(stdscr, 3+i, 2, f"- {filenames[idx]}")
                
                self.safe_addstr(stdscr, 8, 2, "Generating response...")
                stdscr.refresh()
                
                context = "\n".join(retrieved_docs)[:5000]  # Limit context size
                formatted_prompt = self.current_prompt_template.format(
                    context=context,
                    query=query
                )
                
                # Save prompt for debugging
                with open('prompt_debug.txt', 'w') as f:
                    f.write(formatted_prompt)
                
                command = [
                    LLAMA_CMD, 
                    "-m", self.current_model, 
                    "-p", formatted_prompt, 
                    "-n", "512",  # Increased token count for better answers
                    "--temp", "0.7"
                ]
                
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if process.returncode == 0:
                    response = process.stdout.strip()
                    # Clean up the response to show just the answer
                    if "Answer:" in response:
                        response = response.split("Answer:", 1)[1].strip()
                    self.display_response(stdscr, response)
                else:
                    self.display_error(stdscr, f"Model error: {process.stderr}")
                
            except Exception as e:
                self.display_error(stdscr, f"Error: {str(e)}")
                
        finally:
            curses.noecho()

    def display_response(self, stdscr, response):
        """Display the response in a scrollable window"""
        curses.curs_set(0)
        h, w = stdscr.getmaxyx()
        
        # Split response into screen-width lines
        lines = []
        for line in response.split('\n'):
            while len(line) > 0:
                lines.append(line[:w-4])
                line = line[w-4:]
        
        current_line = 0
        max_lines = len(lines)
        
        while True:
            stdscr.clear()
            
            # Header
            self.safe_addstr(stdscr, 0, 0, "Answer:", curses.A_BOLD)
            self.safe_addstr(stdscr, 1, 0, "-" * min(w-1, 40))
            
            # Display visible lines
            for i in range(h-3):
                if current_line + i < max_lines:
                    self.safe_addstr(stdscr, 2+i, 0, lines[current_line + i])
            
            # Footer
            self.safe_addstr(stdscr, h-1, 0, "↑/↓: Scroll | Q: Quit", curses.A_DIM)
            
            key = stdscr.getch()
            if key == curses.KEY_UP and current_line > 0:
                current_line -= 1
            elif key == curses.KEY_DOWN and current_line < max_lines - (h-3):
                current_line += 1
            elif key == ord('q') or key == ord('Q'):
                break

    def display_error(self, stdscr, error_msg):
        """Display an error message to the user"""
        curses.curs_set(0)
        h, w = stdscr.getmaxyx()
        
        stdscr.clear()
        self.safe_addstr(stdscr, h//2, max(0, (w - len(error_msg))//2), 
                       error_msg, curses.A_BOLD | curses.color_pair(2))
        self.safe_addstr(stdscr, h//2+2, 2, 
                       "Press any key to continue...", curses.A_DIM)
        stdscr.refresh()
        stdscr.getch()

def main(stdscr):
    try:
        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Menu
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Titles
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Info

        logging.info("Application started")
        interface = RAGInterface()
        
        while True:
            result = interface.display_main_menu(stdscr)
            if result is None:
                break
                
    except Exception as e:
        logging.critical(f"Fatal error in main: {str(e)}", exc_info=True)
        raise
    finally:
        logging.info("Application exiting")

if __name__ == "__main__":
    try:
        logging.info(f"Log file location: {LOG_FILE}")
        curses.wrapper(main)
    except Exception as e:
        logging.critical(f"Top-level exception: {str(e)}", exc_info=True)
        print(f"A critical error occurred. Check the log file at {LOG_FILE} for details.")