import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Path to store the FAISS index file
file_path ="faiss_store_openai.pkl"

# Set up Streamlit interface
st.title("Research Tool")  # Title of the Streamlit web app
st.sidebar.title("Article URL's")  # Title of the sidebar

# Initialize OpenAI language model with specific parameters
llm=OpenAI(temperature=0.9,max_tokens=500)

# Create an empty placeholder on Streamlit for later use
main_placeholder = st.empty()

# Initialize an empty list to store URLs
urls=[]

# Loop to get 3 URLs from user input in the sidebar
for i in range(3):
    url=st.sidebar.text_input(f"URL{i+1}")  # Create text input for each URL
    urls.append(url)  # Add the URL to the list

# Create a button on the sidebar to start processing the URLs
process_url_click= st.sidebar.button("Process URL")

# If the button is clicked, execute the following:
if process_url_click:
    # Load data from the provided URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Started......")
    data = loader.load()

    # Split the loaded data into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],  # Define separators for splitting
        chunk_size= 1000  # Size of each chunk
    )
    main_placeholder.text("Data Splitting Started......")
    docs = text_splitter.split_documents(data)

    # Create embeddings for the documents and save them in a FAISS index
    embeddings = OpenAIEmbeddings()
    vector_store_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building......")
    time.sleep(2)

    # Save the FAISS index to a file in Pickle format
    with open(file_path, "wb") as f:
        pickle.dump(vector_store_openai, f)

# Set up an input field for user queries
query = main_placeholder.text_input("Question:")

# If a query is entered:
if query:
    # Check if the FAISS index file exists
    if os.path.exists(file_path):
        # Load the FAISS index from the file
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Create a retrieval chain using the loaded language model and FAISS index
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # Process the query and get results
        result = chain({"question":query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])  # Display the answer

        # Display sources of the information, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split sources by newline
            for source in sources_list:
                st.write(source)  # Display each source
