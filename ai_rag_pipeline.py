from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

secret_key = os.getenv("langchain_OpenRouter_key")
chat_model = "nvidia/nemotron-3.5-lightning:free"
embedd_model = "openai/text-embedding-3-small"
base_url = "https://openrouter.ai/api/v1"
collection_name = "ai_docs"
vectors_directory = "./vectors/Marketing_Notes"
pdf_file_directory = "./docs/Marketing_Notes.pdf"

if not secret_key:
    raise ValueError("Key not found")

# connect llm with langchain
llm = ChatOpenAI(model=chat_model, base_url=base_url, api_key=secret_key)

# connect embedding llm with langchin
embedding_llm = OpenAIEmbeddings(
    model=embedd_model, api_key=secret_key, base_url=base_url
)

if not Path(vectors_directory).exists():

    # load Document
    document_loader = PyPDFLoader(pdf_file_directory)
    document = document_loader.load()

    # chunking document
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    chunked_document = splitter.split_documents(document)

    # store document in vector database with chromadb
    vector_store = Chroma.from_documents(
        collection_name=collection_name,
        embedding=embedding_llm,
        persist_directory=vectors_directory,
        documents=chunked_document,
    )

# load vectors
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_llm,
    persist_directory=vectors_directory,
)

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are an accurate and knowledgeable AI assistant that answers questions
using retrieved PDF documents.

### INSTRUCTIONS:

1. Answer the user's question using ONLY the information provided in the context.
2. Do not use your own knowledge or invent facts that are not present in the context.
3. Give a clear, complete, and well-structured answer.
4. Start with a direct answer to the question.
5. Explain important concepts in simple language.
6. Use bullet points or numbered lists when they make the answer easier to understand.
7. Avoid unnecessary repetition and irrelevant information.
8. If the context contains only partial information, explain what information is available and what is missing.
9. If the context does not contain enough information to answer the question, respond:
   "I could not find sufficient information about this topic in the provided documents."
10. Do not invent source names or page numbers.

### SOURCE CITATION RULES:

1. Identify the source from the metadata provided in the context.
2. Include the PDF filename and page number for the information used.
3. If information comes from multiple pages, include all relevant page numbers.
4. Put citations at the end of the relevant paragraph when possible.
5. Finish the response with a Sources section.

### REQUIRED OUTPUT FORMAT:

Answer:
[Give a direct and complete answer.]

Key Points:
- [Important point 1]
- [Important point 2]
- [Important point 3, if applicable]

Sources:
- File: [PDF filename]
  Page: [Page number or page range]

### CONTEXT:
{context}

### USER QUESTION:
{question}

### FINAL ANSWER:
""")

# retriver
retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Output Parser
parser = StrOutputParser()


# Petriver Document Joiner
def retriver_document_joiner(a):
    return "\n\n\n".join(f"source:\n {doc.metadata} \n\n content:\n {doc.page_content}" for doc in a)


# LCLE
chain = (
    {
        "context": retriver | RunnableLambda(retriver_document_joiner),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser
)


while True:
    print("type s or stop to stop chatbot.")
    question = input("~~~> Enter Question: ")

    if question.strip().lower() in ["s", "stop"]:
        print("thank you for using me.")
        break
    print("===> Response: \n")
    for chunk in chain.stream(question):
        print(chunk, end="")
    print("\n\n")
