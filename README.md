# AmbedkarGPT - RAG-based Q&A System

A simple yet powerful command-line Question & Answer system built using the RAG (Retrieval-Augmented Generation) architecture. This system ingests Dr. B.R. Ambedkar's speech text and answers questions based solely on that content.

## Features

- **Local & Free**: Runs entirely on your machine with no API keys or cloud dependencies
- **RAG Architecture**: Combines retrieval and generation for accurate, context-based answers
- **Interactive Mode**: Ask multiple questions in a conversational interface
- **Source Attribution**: Shows which text chunks were used to generate each answer

## Architecture

The system follows a classic RAG pipeline:

1. **Document Loading**: Loads `speech.txt` using LangChain's TextLoader
2. **Text Chunking**: Splits text into manageable chunks with overlap
3. **Embedding Creation**: Generates vector embeddings using HuggingFace's sentence-transformers
4. **Vector Storage**: Stores embeddings in ChromaDB for efficient retrieval
5. **Question Processing**: Retrieves relevant chunks based on semantic similarity
6. **Answer Generation**: Feeds context and question to Ollama's Mistral 7B model

## Technology Stack

- **Framework**: LangChain (orchestration)
- **Vector Database**: ChromaDB (local, persistent storage)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Ollama with Mistral 7B
- **Language**: Python 3.8+

## Prerequisites

1. **Python 3.8 or higher** installed
2. **Ollama** installed and running
3. **Mistral 7B model** pulled in Ollama

## Installation

### Step 1: Install Ollama

**For macOS:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

**For Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Pull Mistral Model
```bash
ollama pull mistral
```

### Step 3: Clone Repository
```bash
git clone https://github.com/yourusername/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Step 4: Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Start Ollama Server

Ensure Ollama is running (check for llama icon in menu bar on macOS)

### Run the Q&A System
```bash
python main.py
```

### Example Interaction
```
==============================================================
AmbedkarGPT Q&A System - Interactive Mode
==============================================================

Your question: What is the real remedy according to the speech?

Answer: The real remedy is to destroy the belief in the sanctity of the shastras.

Your question: quit
Thank you for using AmbedkarGPT!
```

## Project Structure
```
AmbedkarGPT-Intern-Task/
│
├── main.py              # Main application code
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── speech.txt          # Input text (Dr. Ambedkar's speech)
└── chroma_db/          # ChromaDB storage (created on first run)
```

## Troubleshooting

### Issue: "Could not connect to Ollama"
**Solution**: Ensure Ollama is running

### Issue: "Model not found"
**Solution**: Pull the Mistral model with `ollama pull mistral`

### Issue: "speech.txt not found"
**Solution**: Ensure `speech.txt` is in the same directory as `main.py`

## How It Works

1. **Document Ingestion**: The speech text is loaded and split into overlapping chunks
2. **Embedding Generation**: Each chunk is converted into a vector using sentence-transformers
3. **Vector Storage**: Embeddings are stored in ChromaDB with the original text
4. **Query Processing**: Questions are converted to embeddings
5. **Similarity Search**: ChromaDB finds the most similar chunks
6. **Answer Generation**: Retrieved chunks and question are sent to Mistral 7B for answer generation

## Author

Created as part of the AmbedkarGPT Internship Task

