# To be able to run LLMs locally
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# To train them locally
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# For the prompt template
from langchain_core.prompts import ChatPromptTemplate

# For the chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# To traverse a directory
import os

question = "What is Burger Queen ?"

llm = Ollama(
    model="llama3.1", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
# Get the original answer
# answer = llm.invoke(question)

# Step 1: Load some data from a directory. Traverse the wiki directory and load each documents
wiki_path = "wiki"  # Path with all the files you would like to train your LLM on
documents = []
for root, dirs, files in os.walk(top=wiki_path, topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# Step 2: Split the document into chunks with a specified chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(documents)

# Step 3: Store the documents into a vector store with a specific embedding model
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings, persist_directory="./chroma_db") # save to disk

# Step 4: Use RAG (Retrieval-augmented generation)
prompt = ChatPromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | llm
    | StrOutputParser()
)


docs = vectorstore.similarity_search(question)

# Run
chain.invoke({"context": docs, "question": question})



