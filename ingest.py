import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/"
DB_PATH = "vector_store/"


loader_mapping = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

def create_vector_db():
    """
    Creates a vector database from documents in the DATA_PATH.
    """
    print("Loading documents...")
    

    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*", 
        use_multithreading=True,
        show_progress=True,
        loader_kwargs=loader_mapping,

    )
    
    documents = loader.load()
    
    if not documents:
        print("No documents were successfully loaded. Check file types and paths.")
        print("Ensure your 'data' folder contains .pdf, .docx, or .txt files.")
        return

    print(f"Loaded {len(documents)} document chunks.")

    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} text chunks.")

    print("Creating embeddings... (This may take a while)")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
    )

    print("Creating and persisting vector store...")
    db = Chroma.from_documents(texts, embedding_model, persist_directory=DB_PATH)
    print("Vector store created successfully!")

if __name__ == "__main__":
    create_vector_db()