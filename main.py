from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import UnstructuredURLLoader
import nltk
nltk.download('punkt')
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
import streamlit as st
import pickle
import time
from langchain_openai import ChatOpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

load_dotenv()
req_input = os.getenv("REQ_IN")
os.environ["OPENAI_API_KEY"] = req_input


llm = ChatOpenAI()
file_path = ""
st.title("News Analyzer Tool 🔍")
st.write("*News Articles or Online Documents behind a Paywall cannot be processed*")
st.sidebar.write("Examples and Documentation for this Application can be found [here](https://github.com/MukundAravapalli/st_news_analyzer/blob/main/README.md).")
st.sidebar.title("News Article URLs 📰")
number_of_articles = 5

urls = []

for i in range(number_of_articles):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

mukund_link = "https://www.mukund-aravapalli.com/"
tutorial_link = "https://www.youtube.com/watch?v=MoqgmWV1fm8"
dhaval_link = "https://www.linkedin.com/in/dhavalsays"

st.sidebar.markdown(f"This version of the News Analyzer Tool was built by [Mukund Aravapalli]({mukund_link}), but the original application was built by [Dhaval Patel]({dhaval_link}) and is called the News Research Tool. The link for the tutorial can be found [here]({tutorial_link})")


main_placeholder = st.empty()
process_url_clicked = st.button("Enter")
answer_status = st.empty()

if process_url_clicked:
    # load the data
    main_placeholder.text("Loading the data... 🗃️")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    #split the data
    main_placeholder.text("Splitting the data... 📕📗📘📙")
    text_splitter = RecursiveCharacterTextSplitter(
        separators= ["\n\n", "\n", "."],
        chunk_size = 1000,
        chunk_overlap = 200
    )

    docs = text_splitter.split_documents(data)

    #Create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding the data... 📊")
    time.sleep(2)

    #Save the FAISS index to a pickle file

    # storing results of the vector index 
    file_path = "vector_index.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_openai.serialize_to_bytes(), f)
    
    answer_status.write("Loading answer, this can take a few seconds...")

query = main_placeholder.text_input("Question: ")


if query:
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            uploaded_pickle = pickle.load(f)
            vectorIndex = vectorindex_openai.deserialize_from_bytes(serialized=uploaded_pickle, embeddings=embeddings)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
            result = chain.invoke({'question': query}, return_only_outputs = True)
            st.header("Answer")
            st.write(result["answer"])
            answer_status.write("")

            #Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources: ")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    


# Use the following line in terminal to run streamlit
# streamlit run main.py
