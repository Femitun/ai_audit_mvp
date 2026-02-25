import os
import re
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_community.vectorstores import Chroma

# Load environment variables (your Google API Key)
load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "10"))
BATCH_SLEEP_SECONDS = float(os.getenv("INGEST_BATCH_SLEEP_SECONDS", "2"))
MAX_RETRIES = int(os.getenv("INGEST_MAX_RETRIES", "5"))
RESET_CHROMA = os.getenv("RESET_CHROMA", "false").lower() == "true"

def main():
    # 1. Load the documents
    print("Loading documents from the data directory...")
    documents = load_documents()
    if not documents:
        print("No documents found. Please add a PDF to the 'data' folder.")
        return

    # 2. Split the documents into chunks
    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # 3. Save to ChromaDB
    print("Generating embeddings and saving to ChromaDB...")
    save_to_chroma(chunks)
    print("Ingestion complete.")

def load_documents():
    """Reads all PDF files from the data directory."""
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents):
    """Breaks large documents into smaller, 1000-character chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, # Overlap prevents cutting sentences in half
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def save_to_chroma(chunks):
    """Converts chunks to embeddings using Gemini and stores them locally."""
    if RESET_CHROMA and os.path.exists(CHROMA_PATH):
        print(f"Resetting existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    elif os.path.exists(CHROMA_PATH):
        print(f"Updating existing database at {CHROMA_PATH}")

    # Initialize Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    total_chunks = len(chunks)
    print(f"Ingesting {total_chunks} chunks in batches of {BATCH_SIZE}...")

    for start in range(0, total_chunks, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_chunks)
        batch = chunks[start:end]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                db.add_documents(batch)
                print(f"Added chunks {start + 1}-{end}/{total_chunks}")
                break
            except GoogleGenerativeAIError as error:
                message = str(error)
                if "RESOURCE_EXHAUSTED" not in message or attempt == MAX_RETRIES:
                    raise

                retry_match = re.search(r"retry in ([\d.]+)s", message)
                wait_seconds = float(retry_match.group(1)) if retry_match else BATCH_SLEEP_SECONDS * attempt
                wait_seconds = max(wait_seconds, BATCH_SLEEP_SECONDS)
                print(f"Quota reached. Waiting {wait_seconds:.1f}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait_seconds)

        if end < total_chunks:
            time.sleep(BATCH_SLEEP_SECONDS)

if __name__ == "__main__":
    main()