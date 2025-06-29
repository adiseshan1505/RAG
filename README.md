<h1>📄 PDF Bot Version:- 1.1</h1>

<p><strong>RAG for PDFs</strong> is a lightweight AI chat system built for uploading, indexing, and chatting with PDFs using RAG (Retrieval-Augmented Generation). It combines the power of <strong>FastAPI</strong>, <strong>Streamlit</strong>, and <strong>Ollama</strong> (with TinyLlama and nomic Embed Text) to deliver accurate, conversational insights from documents.</p>

<hr>

<h2>🚀 Tech Stack</h2>
<ul>
  <li><strong>FastAPI</strong> - Backend API for managing document ingestion, embeddings, and vector store operations</li>
  <li><strong>Streamlit</strong> - Frontend UI for uploading PDFs and interacting with the AI chatbot</li>
  <li><strong>Ollama</strong> - Running TinyLlama for lightweight local LLM inference and Mnomic for embeddings</li>
</ul>

<hr>

<h2>🧠 Features</h2>
<ul>
  <li>📥 Upload and parse PDFs</li>
  <li>🔍 Create and query vector embeddings using nomic emebed text model</li>
  <li>💬 Chat interface powered by Streamlit</li>
  <li>🧾 Chat history and session management</li>
  <li>🔄 Continue previous conversations anytime</li>
  <li>🤖 Lightweight AI bot using TinyLlama</li>
</ul>

<hr>

<h2>🛠️ Getting Started</h2>

<ol>
  <li>Clone the repository</li>
  <li>Install dependencies</li>
  <li>Start the FastAPI backend</li>
  <li>Run Streamlit frontend</li>
</ol>

<pre>
- git clone https://github.com/adiseshan1505/RAG.git
- cd RAG
- Use python version 3.10.x
- activate venv
    - for windows:- .\.venv\Scripts\Activate.ps1
    - for MacOs or Linux:- source venv/bin/activate
- pip install -r requirements.txt
- install ollama from ollama website if not there
    -add it to your path
    -then pull models:-
        -ollama pull tinyllama:1.1b
        -ollama pull nomic-embed-text
- run Streamlit file from backend directory
    - streamlit run streamlitMain.py
</pre>