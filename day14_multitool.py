from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import chromadb
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiTool_Agent")

# ===== SETUP (For RAG) =====
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection("my_data")

docs = [
    "AK Deep Lens created a documentary on the secrets of aliens and UFOs.",
    "Video about the scary story of Bhangarh Fort.",
    "Documentary on Pakistan's independence in 1947."
]
embeddings = embed_model.encode(docs).tolist()
collection.add(documents=docs, embeddings=embeddings, ids=["1", "2", "3"])

# ===== TOOL 1: RAG (Search Local Data) =====
@tool
def search_local_data(query: str) -> str:
    """Searches for the answer from stored documents (AK Deep Lens scripts)."""
    logger.info(f"RAG search: {query}")
    query_emb = embed_model.encode([query]).tolist()
    result = collection.query(query_embeddings=query_emb, n_results=2)
    return " ".join(result['documents'][0])

# ===== TOOL 2: WEB SEARCH (Search Internet) =====
@tool
def search_internet(query: str) -> str:
    """Searches the internet for the latest information (news, current events)."""
    logger.info(f"Web search: {query}")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=2)
            return " ".join([r['body'] for r in results])
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return "Internet search failed."

# ===== AI + AGENT (2 tools) =====
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)

system_prompt = """You are the research assistant for AK Deep Lens.
- Use 'search_local_data' for internal documents or scripts.
- Use 'search_internet' for the latest news or current events.
- Choose the correct tool based on the user's question.
- Answer in English."""

agent = create_agent(llm, [search_local_data, search_internet], system_prompt=system_prompt)

# ===== EXECUTION =====
def ask_agent(question):
    try:
        result = agent.invoke({"messages": [("user", question)]})
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"Error: {e}")
        return "System is busy."

# Test 1: Question for local data
print("\n===== QUESTION 1 (from local data) =====")
print(ask_agent("What was in the aliens documentary?"))

# Test 2: Question for the internet
print("\n===== QUESTION 2 (from internet) =====")
print(ask_agent("What is the latest news in Pakistan today?"))