#%%
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os, shutil
DATA_PATH = "data/books"
CHROMA_DB_PATH = "chroma"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

settings = Settings(persist_directory=CHROMA_DB_PATH, anonymized_telemetry=False)

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=100,
                                                    length_function=len,
                                                    add_start_index=True
    )
    documents = load_documents()
    chunks = text_splitter.split_documents(documents)
    print(f"{len(documents)} documents split into {len(chunks)} chunks")
    document = chunks[47]
    print(document.page_content)
    print(document.metadata)
    return chunks

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents
#%%
def save_to_chroma(chunks):
    # set up a db for the chunks
    if not os.path.exists(CHROMA_DB_PATH):
        os.mkdir(CHROMA_DB_PATH)
    print("making db")
    # create new db from documents
    db = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL), 
        
        persist_directory=CHROMA_DB_PATH
    )
    print('peristing')
    db.persist()
    print('persisted')
    print(f"Saved {len(chunks)} chunks into {CHROMA_DB_PATH}")

if __name__ == "__main__":
    main()