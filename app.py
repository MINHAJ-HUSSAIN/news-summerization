import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from news_scraper import newsscrapper
import os
import shutil
import time

# Streamlit setup
st.title("News Article Scraper & Query Answering App")
st.sidebar.title("Enter News URLs")
url1 = st.sidebar.text_input("URL 1:")
url2 = st.sidebar.text_input("URL 2:")
url_process = st.sidebar.button("Process URLs")
maintext_placeholder = st.empty()

# Embedding model setup
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
Groq_API_TOKEN = "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"
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=Groq_API_TOKEN,
    temperature=0.7,
    max_tokens=1024
)

vector_database_filepath = "./vectorstore_dataset"

if url_process:
    maintext_placeholder.text("Loading URLs...")
    Url_link1 = [newsscrapper(url1, 1)]
    Url_link2 = [newsscrapper(url2, 2)]
    Url_all_document = [doc for doc in Url_link1 + Url_link2 if doc]

    if Url_all_document:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=0
        )
        Url_textall = []
        for doc in Url_all_document:
            chunk = text_splitter.split_documents([doc])
            Url_textall.extend(chunk)
        
        vectorstore_Huggingface = FAISS.from_documents(Url_textall, embeddings)
        vectorstore_Huggingface.save_local(vector_database_filepath)
        maintext_placeholder.text("Vector Embedding Stored ✔✔✔")
    else:
        maintext_placeholder.text("No valid URLs were provided.")

# Query Input
question = st.text_input("Ask a question:")

if question:
    if os.path.exists(vector_database_filepath):
        vectorstore_load = FAISS.load_local(vector_database_filepath, embeddings)
        retriever = vectorstore_load.as_retriever(search_kwargs={"k": 3})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        response = chain.invoke({"question": question})
        answer = response["answer"]
        sources = response.get("sources", "")

        st.header("Answer")
        st.write(answer)
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        general_answer = llm.predict(f"Answer the question: {question}")
        st.write(general_answer)
