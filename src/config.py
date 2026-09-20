"""
Central configuration for the AI Training Assistant.
Reads environment variables (from .env if present) and defines
project paths, routing settings, and LLM backend selection.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
# Load the project's .env regardless of the directory from which Streamlit is
# launched. Environment variables already set by the host take precedence.
load_dotenv(dotenv_path=ENV_FILE)
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_SET_PATH = DATA_DIR / "evaluation_set.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Knowledge routes -> folders inside data/corpus
# Mirrors the routing paths required by the capstone spec.
# ---------------------------------------------------------------------------
ROUTE_SOURCES = {
    "general_company": [CORPUS_DIR / "company"],
    "role_specific": [CORPUS_DIR / "roles"],
    "admin_policy": [CORPUS_DIR / "policies", CORPUS_DIR / "admin", CORPUS_DIR / "faq"],
    # direct_llm has no document source; it is the fallback path
}

ROUTE_LABELS = ["general_company", "role_specific", "admin_policy", "direct_llm"]

# ---------------------------------------------------------------------------
# Routing / retrieval tuning
# ---------------------------------------------------------------------------
# Minimum TF-IDF cosine similarity a route must achieve to be considered
# "confident". If the best route's score is below this, we fall back to
# direct_llm rather than force an answer from an unrelated knowledge base.
ROUTING_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTING_CONFIDENCE_THRESHOLD", "0.08"))

# Number of chunks retrieved per query
TOP_K = int(os.getenv("TOP_K", "3"))

# ---------------------------------------------------------------------------
# Retrieval backend
#   "tfidf" -> original scikit-learn TF-IDF cosine similarity retriever
#   "faiss" -> dense-embedding retriever backed by a FAISS index
# ---------------------------------------------------------------------------
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "tfidf")  # tfidf | faiss
FAISS_EMBEDDING_DIM = int(os.getenv("FAISS_EMBEDDING_DIM", "128"))

# ---------------------------------------------------------------------------
# Router backend
#   "rule_based" -> keyword + TF-IDF/FAISS blended router (default)
#   "llm"        -> LLM-based query classification
# ---------------------------------------------------------------------------
ROUTER_BACKEND = os.getenv("ROUTER_BACKEND", "rule_based")  # rule_based | llm

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
CONVERSATION_MEMORY_TURNS = int(os.getenv("CONVERSATION_MEMORY_TURNS", "4"))

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
ANALYTICS_DIR = DATA_DIR / "analytics"
ANALYTICS_DIR.mkdir(exist_ok=True)
ANALYTICS_DB_PATH = ANALYTICS_DIR / "analytics.db"

# ---------------------------------------------------------------------------
# LLM backend selection
#   "groq"      -> uses GROQ_API_KEY (OpenAI-compatible API)
#   "extractive"-> no external API call; deterministic template answer
#                  built directly from the retrieved passages (works offline,
#                  used automatically if no API key is configured).
#
# Anthropic and OpenAI backends have been removed from this build. Only
# Groq (for LLM answers) and Serper (for web-search fallback) are used.
# ---------------------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")  # auto | groq | extractive
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Groq's browser search runs server-side.  "auto" enables it when a Groq
# key is configured; set false to use Serper only.
_GROQ_WEB_SEARCH_FLAG = os.getenv("GROQ_WEB_SEARCH_ENABLED", "auto").strip().lower()
if _GROQ_WEB_SEARCH_FLAG == "auto":
    GROQ_WEB_SEARCH_ENABLED = bool(GROQ_API_KEY)
else:
    GROQ_WEB_SEARCH_ENABLED = _GROQ_WEB_SEARCH_FLAG in ("1", "true", "yes", "on") and bool(GROQ_API_KEY)
GROQ_WEB_SEARCH_MODEL = os.getenv("GROQ_WEB_SEARCH_MODEL", "openai/gpt-oss-20b")

ASSISTANT_NAME = "Onboarding Buddy"
COMPANY_NAME = os.getenv("COMPANY_NAME", "Example Organization")

# ---------------------------------------------------------------------------
# Admin dashboard login
#   Simple username/password gate in front of the Admin/Analytics page.
#   Override in .env for anything beyond local/demo use.
# ---------------------------------------------------------------------------
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# ---------------------------------------------------------------------------
# Web search (Serper.dev — Google Search API)
#
# Used as a fallback ONLY when a question doesn't match anything in the
# local onboarding knowledge base (route == "direct_llm"). Rather than
# guessing or refusing, the assistant can pull live web results and answer
# from those instead, clearly citing the source URLs.
#
# Get a free key at https://serper.dev (2,500 free searches, no card
# required), then set SERPER_API_KEY below. Leave it blank to disable web
# search entirely — the assistant works fully without it.
# ---------------------------------------------------------------------------
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_API_URL = os.getenv("SERPER_API_URL", "https://google.serper.dev/search")
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "5"))
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "8"))
# "auto" -> enabled automatically whenever SERPER_API_KEY is set.
# Set WEB_SEARCH_ENABLED=false to keep the key configured but skip web
# search anyway (e.g. temporarily), or "true" to force it on.
_WEB_SEARCH_FLAG = os.getenv("WEB_SEARCH_ENABLED", "auto").strip().lower()
if _WEB_SEARCH_FLAG == "auto":
    WEB_SEARCH_ENABLED = bool(SERPER_API_KEY)
else:
    WEB_SEARCH_ENABLED = _WEB_SEARCH_FLAG in ("1", "true", "yes", "on") and bool(SERPER_API_KEY)
