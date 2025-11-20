import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

SPEECH_FILE = "speech.txt"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

def load_and_split_document(file_path):
    
    print(f"Loading document from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Please ensure the file exists.")
    
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Document split into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks):
    
    print("Creating embeddings using HuggingFace model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("Storing embeddings in ChromaDB...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    vectorstore.persist()
    
    print("Vector store created successfully!")
    return vectorstore

def setup_qa_chain(vectorstore):
    
    print("Setting up Ollama LLM...")
    
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.3  # Lower temperature for more factual responses
    )
    
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always base your answer strictly on the provided context.

Context: {context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
   
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    print("QA chain ready!")
    return qa_chain

def ask_question(qa_chain, question):
   
    print(f"\nQuestion: {question}")
    print("Retrieving relevant context and generating answer...\n")
    
    result = qa_chain({"query": question})
    
    answer = result['result']
    source_docs = result['source_documents']
    
    print(f"Answer: {answer}\n")
    
    print("--- Source Chunks Used ---")
    for i, doc in enumerate(source_docs, 1):
        print(f"Chunk {i}: {doc.page_content[:100]}...")
    print("-" * 50)
    
    return answer

def interactive_mode(qa_chain):
    
    print("\n" + "="*60)
    print("AmbedkarGPT Q&A System - Interactive Mode")
    print("="*60)
    print("Ask questions about the speech. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using AmbedkarGPT!")
                break
            
            if not question:
                print("Please enter a valid question.\n")
                continue
            
            ask_question(qa_chain, question)
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")

def main():
    
    print("="*60)
    print("AmbedkarGPT - RAG Q&A System")
    print("="*60)
    print()
    
    try:
        
        chunks = load_and_split_document(SPEECH_FILE)
       
        vectorstore = create_vector_store(chunks)
        
        qa_chain = setup_qa_chain(vectorstore)
        
        interactive_mode(qa_chain)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure speech.txt is in the same directory as this script.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure Ollama is installed and running: ollama serve")
        print("2. Ensure Mistral model is pulled: ollama pull mistral")
        print("3. Check that all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
