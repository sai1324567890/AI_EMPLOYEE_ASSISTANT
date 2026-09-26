"""
LangChain building blocks for the AI Training Assistant.

- get_chat_model():  Groq via LangChain's OpenAI-compatible ChatOpenAI
- build_chain():     LCEL chain  ChatPromptTemplate | chat model | StrOutputParser
- RouteRetriever:    LangChain BaseRetriever wrapping the existing TF-IDF / FAISS retrievers
- docs_to_retrieved: LangChain Documents -> RetrievedChunk (keeps the rest of the pipeline unchanged)
"""
from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from . import config
from .retriever import RetrievedChunk

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ---------------------------------------------------------------------------
# LLM + chain
# ---------------------------------------------------------------------------
def get_chat_model(model: str = None, max_tokens: int = 600) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        max_tokens=max_tokens,
    )


# Values are passed as variables, so braces inside the prompts (e.g. the
# router's JSON example) are safe.
_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "{system_prompt}"), ("human", "{user_prompt}")]
)


def build_chain(model: str = None, max_tokens: int = 600):
    """prompt | llm | parser  ->  .invoke({"system_prompt": ..., "user_prompt": ...}) -> str"""
    return _PROMPT | get_chat_model(model, max_tokens) | StrOutputParser()


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class RouteRetriever(BaseRetriever):
    """Wraps Retriever / VectorRetriever for one already-chosen route."""

    retriever: Any
    route: str
    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        hits = self.retriever.top_k(query, self.route, k=self.k)
        return [
            Document(
                page_content=h.chunk.text,
                metadata={
                    "citation": h.chunk.citation,
                    "route": h.chunk.route,
                    "score": h.score,
                    "chunk": h.chunk,
                },
            )
            for h in hits
        ]


def docs_to_retrieved(docs: List[Document]) -> List[RetrievedChunk]:
    return [RetrievedChunk(chunk=d.metadata["chunk"], score=d.metadata["score"]) for d in docs]
