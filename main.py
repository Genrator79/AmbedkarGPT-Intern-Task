"""
AmbedkarGPT-Intern-Task - main.py
Simple command-line Q&A using:
- LangChain
- HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
- ChromaDB local vector store
- Ollama LLM (mistral 7b) as local LLM via LangChain's Ollama wrapper

How it works:
1. Loads speech.txt
2. Splits text into chunks
3. Creates / loads embeddings with Chroma
4. Runs a RetrievalQA chain using Ollama as LLM
"""

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def build_or_load_vectorstore(
    text_path: str,
    persist_directory: str = "chromadb",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    force_recreate: bool = False
):
    """
    Loads text, splits, creates embeddings and builds a Chroma vectorstore.
    If a persisted store exists and force_recreate is False, it loads the existing store.
    """
    # create embeddings object
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # if persisted directory exists, load
    should_load = os.path.exists(persist_directory) and not force_recreate
    if should_load:
        print(f"Loading existing Chroma vectorstore from '{persist_directory}'...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore, embeddings

    # otherwise build from scratch
    print("Loading text from:", text_path)
    loader = TextLoader(text_path, encoding="utf-8")
    docs = loader.load()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        respect_sentence_boundary=True
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks.")

    # create Chroma vectorstore and persist it
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print("Vectorstore created and persisted at:", persist_directory)
    return vectorstore, embeddings

def create_retrieval_qa(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # simple chain for small context
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain

def main():
    text_path = "speech.txt"
    persist_dir = "chromadb"

    # Step 1-3: Build or load vectorstore
    vectorstore, embeddings = build_or_load_vectorstore(
        text_path=text_path,
        persist_directory=persist_dir,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=100,
        force_recreate=False
    )

    # Step 4: Configure local Ollama LLM
    # Ensure Ollama daemon is running and Mistral 7B is pulled: `ollama pull mistral`
    # The model name in Ollama is often 'mistral'. Adjust if necessary.
    llm = Ollama(model="mistral", temperature=0.0, max_tokens=512)

    # Step 5: Build QA chain
    qa = create_retrieval_qa(vectorstore, llm)

    print("\nAmbedkarGPT — Ask questions about the provided speech (type 'exit' to quit)\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break
        if not query:
            continue

        # run the chain
        try:
            result = qa.run(query)
            print("\nAnswer:\n", result.strip(), "\n")
        except Exception as e:
            print("Error while running QA chain:", e)
            print("Make sure Ollama is running and the Mistral model is available.")
            break

if __name__ == "__main__":
    main()
