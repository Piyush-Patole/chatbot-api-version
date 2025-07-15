import os      #To access local files and folders
import streamlit as st    #For UI
from dotenv import load_dotenv    #To access .env for Tokens/API Keys

from langchain.memory import ConversationBufferMemory      #To get the previous chat context
from langchain.chains import ConversationChain, RetrievalQA      #ConvChain when any files not Uploaded. Retreival for RAG(Files uploaded case)
from langchain.prompts import PromptTemplate        #To deign the prompt and questions for model

from langchain_community.vectorstores import FAISS    #To efficiently perform searching and clustering in dense vectors
from langchain_community.embeddings import HuggingFaceHubEmbeddings  #To make embeddings from input text
from langchain.text_splitter import RecursiveCharacterTextSplitter    #Breaks long text or document into pieces
from langchain_community.document_loaders import PyPDFLoader, TextLoader        #To load and pdfs and texts

from langchain_community.llms import HuggingFaceHub  #For HuggingFaces

load_dotenv() 

# Initialize LLM using Hugging Face; Token fetched from env; Temperature sets creativity(0-1); max tokens sets response limit
def init_llm():
    if "llm" not in st.session_state:
        st.session_state.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_kwargs={"temperature": 0.7, "max_new_tokens": 400}
        )
    return st.session_state.llm

# Conversation memory; Keeps history
def init_memory():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
    return st.session_state.memory

# Load and embed documents; Uploaded files are stored in temp folder; once text is loaded File is deleted later; Big doc is saved into small chunks and later converted into vector using embeddings and stored for searching in QA; Returns true if worked
def process_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_path = f"./temp/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file_path) if file.name.endswith(".pdf") else TextLoader(file_path)
        documents.extend(loader.load())
        os.remove(file_path)

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceHubEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

        st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
        return True
    return False

# Streamlit UI; 
def main():
    st.title("Your Newbie Chatbot")
    st.caption("Powered by free-tier Hugging Face API")

    with st.sidebar:
        st.subheader("Uploading Documents")
        uploaded_files = st.file_uploader(
            "Upload yours PDFs or text files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            with st.spinner("Processing..."):
                if process_documents(uploaded_files):
                    st.success("Documents loaded successfully!")
                else:
                    st.error("Document loading failed")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                if "vectorstore" in st.session_state:
                    retriever = st.session_state.vectorstore.as_retriever()
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=init_llm(),
                        chain_type="stuff",
                        retriever=retriever,
                        memory=init_memory()
                    )
                    response = qa_chain.invoke({"query": prompt})["result"]
                else:
                    conversation = ConversationChain(
                        llm=init_llm(),
                        memory=init_memory(),
                        prompt=PromptTemplate(
                            input_variables=["history", "input"],
                            template="""
                            You are a helpful assistant.
                            
                            History: {history}
                            Question: {input}
                            Answer:"""
                        )
                    )
                    response = conversation.run(prompt)

                for chunk in response.split():
                    full_response += chunk + " "
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            except Exception as e:
                full_response = "Error: " + str(e)
                st.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.makedirs("temp")
    main()