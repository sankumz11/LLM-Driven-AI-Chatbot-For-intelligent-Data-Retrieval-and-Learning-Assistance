import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
import edge_tts
import tempfile
import asyncio
import os

# --- API Configuration ---
GOOGLE_API_KEY = "#enter your api key"

# --- Embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    google_api_key=GOOGLE_API_KEY
)

# --- Model Config ---
CHAT_MODEL = "gemini-1.5-pro-001"

# --- PDF Processing ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"PDF Error: {str(e)}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)

# --- Vector Store (ChromaDB) ---
def get_vector_store(text_chunks):
    try:
        # Remove old DB to avoid dimension mismatch errors
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
        vector_store = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        st.session_state.vector_store = vector_store
        st.success("Documents processed and stored in ChromaDB!")
    except Exception as e:
        st.error(f"Vector Error: {str(e)}")
        st.stop()

# --- Conversation Setup ---
def get_conversation_chain():
    prompt_template = """
Use context and chat history to answer. If unsure, say "Not in documents."

Context:
{context}

History:
{chat_history}

Question: 
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    # Use InMemoryChatMessageHistory for session-based chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()

    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding=embeddings
    )

    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.chat_history,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )}
    )

# --- Quiz Generation ---
def generate_quiz():
    try:
        qa = get_conversation_chain()
        response = qa({"question": "Generate three multiple-choice quiz questions based on the documents."})
        return response["answer"]
    except Exception as e:
        st.error(f"Quiz Generation Error: {str(e)}")
        return ""

# Text to Speech 
async def text_to_speech(text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            await communicate.save(temp_audio.name)
            st.audio(temp_audio.name)
    except Exception as e:
        st.error(f"Audio Error: {str(e)}")

#  Core Logic
def handle_user_input(question):
    try:
        qa = get_conversation_chain()
        response = qa({"question": question})
        return response["answer"]
    except Exception as e:
        st.error(f"Query Error: {str(e)}")
        return "Error processing query"

# UI Setup 
def main():
    st.set_page_config(page_title="DocAI Pro", layout="wide")
    st.header("Intelligent Document Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.title("Document Controls")
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Analyzing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        get_vector_store(get_text_chunks(raw_text))
                        st.session_state.processed = True
                    else:
                        st.error("No text found")
            else:
                st.warning("Upload PDFs first")

        if st.button("Generate Quiz") and st.session_state.processed:
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Quiz:\n\n{quiz}"
                })

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask about your documents:")
    if user_question:
        if not st.session_state.processed:
            st.warning("Process documents first")
            return

        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.spinner("Analyzing..."):
            response = handle_user_input(user_question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            st.markdown(response)
            asyncio.run(text_to_speech(response))

if __name__ == "__main__":
    main()

