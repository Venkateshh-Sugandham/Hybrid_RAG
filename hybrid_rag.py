import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import os
from nltk.tokenize import word_tokenize
import nltk
from dotenv import load_dotenv

nltk.download('punkt')
load_dotenv()

index_name = "hybrid-search-langchain-pinecone"
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

# Create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit
st.title("Conversational Hybrid RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# Input the Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if Groq API key is provided
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Chat interface
session_id = st.text_input("Session ID", value="default_session")

# Statefully manage chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)

# Process uploaded PDFs
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name

        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)

    # Assuming you have a list of tokenized documents
    corpus = [doc.page_content for doc in documents]

    # Initialize the BM25Encoder
    bm25_encoder = BM25Encoder()

    # Fit the encoder with your tokenized documents
    bm25_encoder.fit(corpus)

    # Store the fitted values to a JSON file
    bm25_encoder.dump("bm25_values.json")

    # Load the encoder from the JSON file
    bm25_encoder = BM25Encoder().load("bm25_values.json")
    

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    dense_embeddings = [embeddings.embed_query(doc.page_content) for doc in splits]

    pinecone_vectors = []
    for i, (doc, dense_vector) in enumerate(zip(splits, dense_embeddings)):
        vector = {
        "id": f"doc_{i}",  # Unique ID for each document chunk
        "values": dense_vector,  # Dense embedding vector
        "metadata": {"text": doc.page_content,"context": doc.page_content[:200]}  # Optional: add metadata for filtering later
        }
        pinecone_vectors.append(vector)

    # Upsert vectors into Pinecone index
    index.upsert(vectors=pinecone_vectors)

    # Set up retriever using Pinecone
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

    # Define prompts
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)
