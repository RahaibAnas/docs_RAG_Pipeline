# PDF-Based RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **LangChain, ChromaDB, and OpenRouter**. It allows users to ask questions about PDF documents and generates answers using relevant information retrieved from those documents.

## Overview

This project implements a complete RAG pipeline:

1. Loads PDF documents.
2. Splits documents into smaller text chunks.
3. Converts chunks into vector embeddings.
4. Stores embeddings in ChromaDB.
5. Retrieves relevant chunks based on user questions.
6. Passes retrieved context to an LLM.
7. Generates answers using the provided document context.
8. Streams responses to the terminal.

The vector database is persistent, so documents do not need to be embedded again every time the application runs, provided the existing collection is available.

## Architecture

```text
                 PDF Document
                      |
                      v
                PyPDFLoader
                      |
                      v
              Text Splitter
                      |
                      v
               Text Chunks
                      |
                      v
              OpenAI Embeddings
                via OpenRouter
                      |
                      v
                 ChromaDB
                      |
                      |
              User Question
                      |
                      v
                  Retriever
                      |
                      v
             Relevant Documents
                      |
                      v
                Prompt Template
                      |
                      v
                ChatOpenAI
                via OpenRouter
                      |
                      v
                Output Parser
                      |
                      v
              Streaming Answer
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | LLM application framework |
| OpenRouter | Access to chat and embedding models |
| ChatOpenAI | Connects to the chat model |
| OpenAIEmbeddings | Generates vector embeddings |
| ChromaDB | Stores and searches vector embeddings |
| PyPDFLoader | Loads PDF documents |
| RecursiveCharacterTextSplitter | Splits documents into chunks |
| LCEL | Connects different components into a chain |
| python-dotenv | Loads API credentials from environment variables |

## Project Structure

```text
project/
│
├── docs/
│   └── xyz.pdf
│
├── vectors/
│   └── xyz/
│       └── ChromaDB files
│
├── .env
├── .gitignore
├── main.py
└── README.md
```

*The structure above is an example; adjust filenames and folders to match your actual project.*

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -U langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma chromadb pypdf python-dotenv
```

## Environment Configuration

Create a `.env` file in the project root:

```env
langchain_OpenRouter_key=YOUR_OPENROUTER_API_KEY
```

Replace the placeholder with your OpenRouter API key.

**Security:** Never upload your `.env` file or expose your API key on GitHub.

## Model Configuration

The project uses the following models through OpenRouter:

**Chat model:**

```python
chat_model = "nvidia/nemotron-3.5-lightning:free"
```

**Embedding model:**

```python
embedd_model = "openai/text-embedding-3-small"
```

**OpenRouter API endpoint:**

```python
base_url = "https://openrouter.ai/api/v1"
```

Model availability and endpoint support can change. Verify that both selected models are accessible through your OpenRouter account.

## How to Run

1. Place your PDF file inside the `docs/` directory.
2. Update the PDF path in the Python application.
3. Configure your API key in `.env`.
4. Run the application:

```bash
python main.py
```

5. Enter a question when prompted.
6. Type `s` or `stop` to exit the chatbot.

## Key Features

### 1. PDF Document Loading

Uses `PyPDFLoader` to extract text and metadata from PDF documents.

### 2. Text Chunking

Uses `RecursiveCharacterTextSplitter` with:

```python
chunk_size = 500
chunk_overlap = 30
```

Chunking helps retrieve relevant sections instead of processing the entire PDF for every question.

### 3. Vector Embeddings

Uses the embedding model to convert document chunks into numerical vector representations.

### 4. Persistent Vector Storage

Uses ChromaDB to store document embeddings locally.

```python
persist_directory = "./vectors/xyz"
```

This allows the application to reuse previously indexed documents.

### 5. Semantic Retrieval

Uses a LangChain retriever:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

The retriever searches for the three most relevant document chunks.

### 6. Context-Aware Answer Generation

The retrieved chunks are passed to the LLM through a prompt template. The prompt instructs the model to answer using the supplied context and acknowledge when the answer is unavailable.

### 7. Streaming Responses

Uses LangChain's `stream()` method to display the generated response incrementally.

## Important Notes

- The application uses a local persistent ChromaDB directory.
- Keep the same embedding model and compatible vector dimensions when querying an existing collection.
- If you change the source PDF, you must update or rebuild the vector database to include the new document content.
- The current application uses similarity search with `k=3`.
- Source metadata is included in the retrieved context, but generated citations should be checked against the original PDF.
- The chatbot can still produce incorrect answers; retrieved context does not guarantee factual accuracy.


## Learning Outcomes

Through this project, the following GenAI concepts are practiced:

- Retrieval-Augmented Generation (RAG)
- Document loading and preprocessing
- Text chunking and overlap
- Vector embeddings
- Vector databases
- Semantic similarity search
- Retrievers
- Prompt engineering
- LangChain Expression Language (LCEL)
- LLM integration through OpenRouter
- Streaming LLM responses

## License

This project is intended for educational and learning purposes.