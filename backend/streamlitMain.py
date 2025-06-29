import streamlit as st
import requests
import json
from typing import List, Dict, Optional
import time
import uuid
import os
import subprocess
import threading
import sys
from datetime import datetime
import socket
import atexit

API_BASE_URL = "http://127.0.0.1:8000"
FASTAPI_PROCESS = None

st.set_page_config(
    page_title="RAG PDF Chat System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        margin-left: 2rem;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        border-left: 4px solid #4caf50;
        margin-right: 2rem;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
        padding: 0.3rem;
        background-color: rgba(255,255,255,0.7);
        border-radius: 0.3rem;
    }
    .status-healthy {
        color: #4caf50;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #f44336;
        font-weight: bold;
    }
    .startup-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_fastapi_server():
    global FASTAPI_PROCESS
    if FASTAPI_PROCESS is None or FASTAPI_PROCESS.poll() is not None:
        try:
            if os.path.exists("fastAPI.py"):
                FASTAPI_PROCESS = subprocess.Popen([
                    sys.executable, "fastAPI.py"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            elif os.path.exists("main.py"):
                FASTAPI_PROCESS = subprocess.Popen([
                    sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Failed to start server: {e}")
            return False
    return True


def stop_fastapi_server():
    global FASTAPI_PROCESS
    if FASTAPI_PROCESS and FASTAPI_PROCESS.poll() is None:
        FASTAPI_PROCESS.terminate()
        FASTAPI_PROCESS.wait()


atexit.register(stop_fastapi_server)


class RAGChatAPI:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def check_health(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def upload_pdf(self, file) -> Dict:
        try:
            files = {"file": (file.name, file.getvalue(), "application/pdf")}
            response = requests.post(f"{self.base_url}/upload", files=files, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": "Upload timeout - file may be too large"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Upload failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def chat_with_rag(self, message: str, session_id: str) -> Dict:
        try:
            payload = {
                "message": message,
                "session_id": session_id
            }
            response = requests.post(f"{self.base_url}/chat", json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": "Chat request timeout"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Chat failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    def get_documents(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/documents", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_chat_history(self, session_id: str) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/sessions/{session_id}/history", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def clear_chat_history(self, session_id: str) -> Dict:
        try:
            response = requests.delete(f"{self.base_url}/sessions/{session_id}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def delete_document(self, filename: str) -> Dict:
        try:
            response = requests.delete(f"{self.base_url}/documents/{filename}", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


api = RAGChatAPI(API_BASE_URL)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "server_started" not in st.session_state:
    st.session_state.server_started = False

if "startup_attempted" not in st.session_state:
    st.session_state.startup_attempted = False


def display_chat_message(role: str, content: str, sources: List[str] = None, timestamp: str = None):
    timestamp_str = ""
    if timestamp:
        timestamp_str = f"<small style='color: #999; float: right;'>{timestamp}</small>"

    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong> {timestamp_str}
            <div style="clear: both; margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        source_text = ""
        if sources and len(sources) > 0:
            source_links = []
            for source in sources:
                if source.strip():
                    source_links.append(f"📄 {source}")
            if source_links:
                source_text = f"<div class='source-info'>📚 Sources: {' | '.join(source_links)}</div>"

        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong> {timestamp_str}
            <div style="clear: both; margin-top: 0.5rem;">{content}</div>
            {source_text}
        </div>
        """, unsafe_allow_html=True)


def check_and_start_server():
    if is_port_in_use(8000):
        st.session_state.server_started = True
        return True

    if not st.session_state.startup_attempted:
        st.session_state.startup_attempted = True
        return start_fastapi_server()

    return False


def main():
    st.markdown('<h1 class="main-header">📚 RAG PDF Chat System</h1>', unsafe_allow_html=True)

    if not st.session_state.server_started:
        st.markdown("""
        <div class="startup-container">
            <h2>🚀 Starting RAG System...</h2>
            <p>Initializing backend server and dependencies</p>
        </div>
        """, unsafe_allow_html=True)

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        with status_placeholder.container():
            st.info("🔍 Checking for existing server...")
            progress_bar.progress(25)
            time.sleep(1)

            if is_port_in_use(8000):
                st.success("✅ Found running server!")
                st.session_state.server_started = True
                progress_bar.progress(100)
                time.sleep(1)
                st.rerun()
            else:
                st.info("🚀 Starting FastAPI server...")
                progress_bar.progress(50)

                if check_and_start_server():
                    st.info("⏳ Waiting for server to initialize...")
                    progress_bar.progress(75)

                    for i in range(10):
                        time.sleep(2)
                        if is_port_in_use(8000):
                            st.success("✅ Server started successfully!")
                            st.session_state.server_started = True
                            progress_bar.progress(100)
                            time.sleep(1)
                            st.rerun()
                            return
                        progress_bar.progress(75 + (i * 2))

                    st.error("❌ Server failed to start in time")
                    progress_bar.progress(100)
                else:
                    st.error("❌ Could not start server automatically")
                    progress_bar.progress(100)

        if not st.session_state.server_started:
            st.markdown("""
            ### Manual Setup Required

            The server could not be started automatically. Please:

            1. **Start FastAPI manually:**
            ```bash
            python fastAPI.py
            ```

            2. **Or use uvicorn:**
            ```bash
            uvicorn main:app --host 127.0.0.1 --port 8000
            ```

            3. **Make sure Ollama is running:**
            ```bash
            ollama serve
            ollama pull nomic-embed-text
            ollama pull tinyllama:1.1b
            ```
            """)

            if st.button("🔄 Check Again", type="primary"):
                st.session_state.startup_attempted = False
                st.rerun()

            return

    health_status = api.check_health()

    if health_status.get("status") == "error":
        st.error("❌ Backend server not responding!")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Connection"):
                st.rerun()
        with col2:
            if st.button("🚀 Restart Server"):
                st.session_state.server_started = False
                st.session_state.startup_attempted = False
                stop_fastapi_server()
                st.rerun()

        return

    with st.sidebar:
        st.header("🔧 System Status")

        if health_status.get("status") == "healthy":
            st.markdown('<div class="metric-card"><p class="status-healthy">✅ System Operational</p></div>',
                        unsafe_allow_html=True)

            ollama_status = health_status.get("ollama", "unknown")
            vector_db_status = health_status.get("vector_db", "unknown")

            col1, col2 = st.columns(2)
            with col1:
                if ollama_status == "connected":
                    st.markdown("🟢 **Ollama**<br>Connected", unsafe_allow_html=True)
                else:
                    st.markdown("🔴 **Ollama**<br>Disconnected", unsafe_allow_html=True)

            with col2:
                st.markdown(f"🟢 **Vector DB**<br>{vector_db_status.title()}", unsafe_allow_html=True)

            if ollama_status != "connected":
                st.warning("⚠️ Start Ollama: `ollama serve`")
        else:
            st.markdown('<div class="metric-card"><p class="status-unhealthy">❌ System Issues</p></div>',
                        unsafe_allow_html=True)

        st.divider()

        st.header("📁 Document Management")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with",
            key="pdf_uploader"
        )

        if uploaded_file is not None:
            if st.button("🚀 Upload & Process", type="primary", key="upload_btn"):
                with st.spinner("Processing PDF..."):
                    result = api.upload_pdf(uploaded_file)

                    if "error" in result:
                        st.error(f"Upload failed: {result['error']}")
                    else:
                        st.success(f"✅ {result.get('message', 'Success!')}")
                        if 'chunks_created' in result:
                            st.info(f"📊 Created {result['chunks_created']} chunks")
                        time.sleep(1)
                        st.rerun()

        st.subheader("📚 Uploaded Documents")
        docs_result = api.get_documents()

        if "error" not in docs_result and docs_result.get("documents"):
            for i, doc in enumerate(docs_result["documents"]):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {doc}")
                with col2:
                    if st.button("🗑️", key=f"delete_{i}", help=f"Delete {doc}"):
                        delete_result = api.delete_document(doc)
                        if "error" not in delete_result:
                            st.success("Deleted!")
                            st.rerun()
                        else:
                            st.error("Delete failed")
        else:
            st.info("No documents uploaded")

        st.divider()

        st.header("💬 Chat Session")
        st.write(f"**Session:** `{st.session_state.session_id[:8]}...`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", key="clear_btn"):
                api.clear_chat_history(st.session_state.session_id)
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            if st.button("🆕 New", key="new_session_btn"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.rerun()

    st.header("💭 Chat Interface")

    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                display_chat_message(
                    message["role"],
                    message["content"],
                    message.get("sources", []),
                    message.get("timestamp")
                )
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 1rem; margin: 2rem 0;">
                <h3>👋 Welcome to RAG PDF Chat!</h3>
                <p>Upload a PDF document and start asking questions.</p>
                <p><strong>Try asking:</strong> "What is this document about?" or "Summarize the main points"</p>
            </div>
            """, unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input(
                "Ask a question...",
                placeholder="Type your question here...",
                key="chat_input",
                label_visibility="collapsed"
            )

        with col2:
            send_button = st.form_submit_button("Send", type="primary", use_container_width=True)

        if send_button and user_input.strip():
            if not docs_result.get("documents"):
                st.warning("⚠️ Upload a PDF first!")
            else:
                user_message = {
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M")
                }
                st.session_state.chat_history.append(user_message)

                with st.spinner("🤔 Thinking..."):
                    response = api.chat_with_rag(user_input, st.session_state.session_id)

                    if "error" in response:
                        error_message = {
                            "role": "assistant",
                            "content": f"❌ Error: {response['error']}",
                            "timestamp": datetime.now().strftime("%H:%M")
                        }
                        st.session_state.chat_history.append(error_message)
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": response.get("response", "No response"),
                            "sources": response.get("sources", []),
                            "timestamp": datetime.now().strftime("%H:%M")
                        }
                        st.session_state.chat_history.append(assistant_message)

                st.rerun()


if __name__ == "__main__":
    main()