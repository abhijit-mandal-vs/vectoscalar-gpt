import asyncio
import random
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import shutil
from vsgpt.retriever import create_retriever
from vsgpt.chain import ask_question, create_chain
from vsgpt.config import Config
from vsgpt.model import create_llm
from vsgpt.ingestor import Ingestor

# Load environment variables
load_dotenv()

LOADING_MESSAGES = [
    "Searching through our knowledge base...",
    "Analyzing relevant documents...",
    "Connecting information from multiple sources...",
    "Retrieving context-specific data...",
    "Synthesizing insights from company documents...",
    "Extracting key information from our database...",
    "Combining relevant data points...",
    "Processing organizational knowledge...",
    "Accessing secure company archives...",
    "Integrating information across departments...",
]

# Helper function to clean up temporary files
def cleanup_temp_dir(temp_dir):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

@st.cache_resource(show_spinner=False)
def load_vector_db():
    vector_db_path = Config.Path.DATABASE_DIR / "chroma_db"
    if Path(vector_db_path).exists():
        try:
            llm = create_llm()
            retriever = create_retriever(llm, vector_store=vector_db_path)
            return create_chain(llm, retriever)
        except Exception as e:
            st.error(f"Error loading VectorDB: {str(e)}")
            st.stop()
    else:
        st.warning("VectorDB not found. Please upload documents to create one.")
        st.stop()

@st.cache_data(show_spinner=False, ttl=60)
def get_latest_config_values():
    vector_db_path = Config.Path.DATABASE_DIR / "chroma_db"
    last_trained_file = Config.Path.DATABASE_DIR / "last_trained.txt"

    if vector_db_path.exists():
        last_trained = last_trained_file.read_text().strip() if last_trained_file.exists() else "Unknown"
        files_count = sum(1 for _ in vector_db_path.rglob('*') if _.is_file())
        Config.Retriever.update_last_trained(last_trained)
        Config.Retriever.update_files_ingested_count(files_count)
    else:
        last_trained = "Not trained yet"
        Config.Retriever.update_files_ingested_count(0)

    return {
        "last_trained": last_trained,
        "files_ingested": Config.Retriever.get_files_ingested_count(),
    }

async def ask_chain(question: str, chain):
    full_response = ""
    assistant = st.chat_message(
        "assistant", avatar=str(Config.Path.IMAGES_DIR / "assistant-avatar.png")
    )
    with assistant:
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        documents = []
        async for event in ask_question(chain, question, session_id="session-id-42"):
            if isinstance(event, str):
                full_response += event
                message_placeholder.markdown(full_response)
            elif isinstance(event, list):
                documents.extend(event)
        if Config.Retriever.SHOW_SOURCES:
            for i, doc in enumerate(documents):
                with st.expander(f"Source #{i+1}"):
                    st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

def show_message_history():
    for message in st.session_state.messages:
        role = message["role"]
        avatar_path = (
            Config.Path.IMAGES_DIR / "assistant-avatar.png"
            if role == "assistant"
            else Config.Path.IMAGES_DIR / "user-avatar.png"
        )
        with st.chat_message(role, avatar=str(avatar_path)):
            st.markdown(message["content"])

async def show_chat_input(chain):
    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=str(Config.Path.IMAGES_DIR / "user-avatar.png")):
            st.markdown(prompt)
        await ask_chain(prompt, chain)

def sidebar_ingestion():
    st.sidebar.title("📁 Document Ingestion Portal")
    st.sidebar.subheader("Upload your documents to update the VectorDB")

    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.sidebar.button("Process Files to DB"):
        with st.spinner("Creating the vector database..."):
            try:
                temp_dir = Path("temp_docs")
                temp_dir.mkdir(parents=True, exist_ok=True)

                file_paths = [temp_dir / file.name for file in uploaded_files]
                for file, path in zip(uploaded_files, file_paths):
                    with open(path, "wb") as f:
                        f.write(file.read())

                # Ingest the documents
                ingestor = Ingestor()
                ingestor.ingest(file_paths)

                # Save last trained timestamp
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                (Config.Path.DATABASE_DIR / "last_trained.txt").write_text(current_time)
                Config.Retriever.update_last_trained(current_time)

                # Update the ingested files count
                files_count = sum(1 for _ in (Config.Path.DATABASE_DIR / "chroma_db").rglob('*') if _.is_file())
                Config.Retriever.update_files_ingested_count(files_count)

                cleanup_temp_dir(temp_dir)  # Clean up temp files
                st.sidebar.success("VectorDB created successfully ✅")
                st.experimental_rerun()  # Reload the app

            except Exception as e:
                st.sidebar.error(f"An error occurred: {str(e)}")

# Initialize Streamlit app and hide Streamlit’s default headers
st.set_page_config(page_title="VectoScalar-GPT", page_icon="🆚")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Display sidebar ingestion UI
sidebar_ingestion()

# Initialize session state and load vector DB
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm VS-Bot 🤖. How can I assist you today?"}]
chain = load_vector_db()

# Display latest config values
latest_config = get_latest_config_values()
st.markdown(f"# :rainbow[VectoScalar-GPT] 🆚🤖")
col1, col2, col3 = st.columns([1, 1.3, 1])
col1.success(f"Version: {Config.VERSION}")
col2.info(f"Last Trained: {latest_config['last_trained']}")
# col3.warning(f"Files Ingested: {latest_config['files_ingested']}")
col3.warning(f"VectorDB: Ready ✅")

# Show chat history and input interface
show_message_history()
asyncio.run(show_chat_input(chain))
