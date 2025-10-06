import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, "..", "documents")
CHROMA_DB_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    print("Starting ingestion process...")
    # 1. Load documents
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_DIR, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    
    if not documents:
        print("No PDF documents found. Exiting.")
        return

    print(f"Loaded {len(documents)} pages from PDFs.")

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # 3. Create embeddings (using the new class)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 4. Create and persist the vector store
    print(f"Creating and persisting vector store at: {CHROMA_DB_DIR}")
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=CHROMA_DB_DIR
    )

    print("Ingestion complete!")

if __name__ == "__main__":
    main()