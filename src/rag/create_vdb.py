from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

GROUP_2_RAG = 'group-2-rag.txt'
OPEN_AI_API_KEY = None # TODO: Fill with API Key when provided

# Creates a vector database that we can use for internal documents.
def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents(GROUP_2_RAG)
    chunks = split_text(documents)
    save_to_chroma(chunks,GROUP_2_RAG)

def load_documents(path:str):
    loader = DirectoryLoader(path, glob="*.txt")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    document = chunks[0]
    print(document.page_content)
    print(document.metadata)
    return chunks

def save_to_chroma(chunks: list[Document], database: str):
    # Clear out the database first.
    if os.path.exists(database):
        shutil.rmtree(database)
    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=OPEN_AI_API_KEY), persist_directory=database
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {database}.")

if __name__ == "__main__":
    main()
