# To be able to run LLMs locally
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# To load vectorsize
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# For the prompt template
from langchain_core.prompts import ChatPromptTemplate

# For the chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

question = "What is Burger Queen ?"

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# load from disk
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=local_embeddings)


llm = Ollama(
    model="llama3.1", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)


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