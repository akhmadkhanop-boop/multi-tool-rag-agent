# 🕵️‍♂️ AK Deep Lens Research Agent

An autonomous, multi-tool AI research assistant built specifically for documentary research, script analysis, and content generation. Powered by **LangChain** and **Groq** (Llama 3.3 70B), this agent intelligently routes queries between a local knowledge base and the live internet to uncover deep insights.

## 🚀 Features
* **Hybrid Search Capabilities:** Seamlessly switches between querying internal script databases and fetching live data from the web.
* **Local RAG (Retrieval-Augmented Generation):** Uses **ChromaDB** and **SentenceTransformers** (`all-MiniLM-L6-v2`) to store and retrieve historical documentary scripts locally and privately.
* **Live Web Search:** Integrates **DuckDuckGo Search** to pull real-time news and facts when local data is insufficient.
* **Agentic Routing:** The LLM autonomously decides which tool (`search_local_data` or `search_internet`) to use based on the context of the user's question.

## 🛠️ Tech Stack
* **Framework:** LangChain / LangGraph
* **LLM:** ChatGroq (Llama-3.3-70b-versatile)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **Web Search:** DuckDuckGo Search API (`duckduckgo-search`)
* **Environment:** Python / python-dotenv

## ⚙️ Setup & Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/ak-deep-lens-research-agent.git](https://github.com/your-username/ak-deep-lens-research-agent.git)
cd ak-deep-lens-research-agent
