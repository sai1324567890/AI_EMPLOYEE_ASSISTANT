"""
Streamlit chatbot UI for the AI Training Assistant.

Run with:
    streamlit run app.py

See also `pages/1_Admin_Dashboard.py` for the analytics dashboard
(frequently asked questions, user activity, query category stats,
and feedback results).
"""
import csv
import uuid

import streamlit as st

from src.assistant import TrainingAssistant
from src import config, analytics, ui, voice

st.set_page_config(
    page_title=config.ASSISTANT_NAME,
    page_icon=ui.page_icon(),
    layout="centered",
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
ui.inject_css(st.session_state["theme"])
ui.render_sidebar_logo()

ROUTE_LABELS_DISPLAY = {
    "general_company": "🏢 General company knowledge base",
    "role_specific": "🧑‍💻 Role & team documents",
    "admin_policy": "📋 Policy / admin documents",
    "web_search": "🌐 Live web search (Serper)",
    "direct_llm": "💬 Direct assistant response (no document/web match)",
}

EXAMPLE_GROUP_LABELS = {
    "general_company": "🏢 General company",
    "role_specific": "🧑‍💻 Role & team",
    "admin_policy": "📋 Policy / admin",
    "direct_llm": "💬 Direct / edge cases",
}


@st.cache_data(show_spinner=False)
def load_example_questions() -> dict:
    """Loads every question from data/evaluation_set.csv, grouped (and
    ordered) by its expected_route, for the sidebar's suggested-questions
    list. Falls back to an empty dict if the file is missing."""
    path = config.EVAL_SET_PATH
    groups: dict = {}
    if not path.exists():
        return groups
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            route = (row.get("expected_route") or "other").strip()
            question = (row.get("question") or "").strip()
            if not question:
                continue
            groups.setdefault(route, []).append(question)
    return groups


EXAMPLE_QUESTION_GROUPS = load_example_questions()


@st.cache_resource(show_spinner="Building knowledge base index...")
def get_assistant(retrieval_backend: str, router_backend: str) -> TrainingAssistant:
    return TrainingAssistant(retrieval_backend=retrieval_backend, router_backend=router_backend)


def _new_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["session_id"] = str(uuid.uuid4())
    get_assistant.clear()


if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    # --- Brand + theme toggle -------------------------------------------
    top_col1, top_col2 = st.columns([3, 2])
    with top_col1:
        st.markdown("### 🧭 Onboarding Buddy")
    with top_col2:
        is_dark = st.toggle(
            "🌙 Dark", value=(st.session_state["theme"] == "dark"), key="ob_theme_toggle"
        )
    new_theme = "dark" if is_dark else "light"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    # --- New chat ---------------------------------------------------------
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        _new_chat()
        st.rerun()

    st.divider()

    # --- Recent chats ---------------------------------------------------------
    # Kept deliberately plain: a simple text list (like a native "Recents"
    # panel), not bordered cards. The active chat gets a subtle background
    # tint (injected below) instead of an icon/emoji marker, and delete is a
    # small low-opacity icon that only stands out on hover.
    st.markdown('<p class="ob-plain-label">Recents</p>', unsafe_allow_html=True)
    recent_convos = analytics.get_recent_conversations(limit=12)
    if recent_convos:
        active_id = st.session_state["session_id"]
        st.markdown(
            f"<style>section[data-testid='stSidebar'] "
            f"[class*='st-key-conv_{active_id}'] button{{"
            f"background: var(--ob-chip-bg); font-weight: 600;}}</style>",
            unsafe_allow_html=True,
        )
        for conv in recent_convos:
            sess_id = conv["session_id"]
            title = conv["title"] or "New chat"
            display_title = title if len(title) <= 34 else title[:34].rstrip() + "…"
            row_col1, row_col2 = st.columns([6, 1])
            with row_col1:
                if st.button(
                    display_title,
                    key=f"conv_{sess_id}",
                    use_container_width=True,
                    help=title,
                ):
                    loaded = analytics.load_conversation(sess_id)
                    if loaded is not None:
                        st.session_state["messages"] = loaded
                        st.session_state["session_id"] = sess_id
                        get_assistant.clear()
                        st.rerun()
            with row_col2:
                if st.button("🗑", key=f"del_{sess_id}", help="Delete this chat"):
                    analytics.delete_conversation(sess_id)
                    if sess_id == st.session_state["session_id"]:
                        _new_chat()
                    st.rerun()
    else:
        st.markdown(
            '<div class="ob-recent-empty">No conversations yet — ask a '
            "question to get started!</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Suggested questions (loaded from the eval set, always visible) -----
    st.markdown('<p class="ob-recent-label">💡 Suggested questions</p>', unsafe_allow_html=True)
    for route, qs in EXAMPLE_QUESTION_GROUPS.items():
        st.caption(EXAMPLE_GROUP_LABELS.get(route, route))
        for i, ex in enumerate(qs):
            if st.button(ex, use_container_width=True, key=f"ex_{route}_{i}"):
                st.session_state["pending_question"] = ex

    st.divider()

    # --- About ---------------------------------------------------------------
    with st.expander("ℹ️ About"):
        st.write(
            "Ask anything about company info, your role, or admin/HR policies. "
            "Questions are routed to the right knowledge source automatically. "
            "The assistant remembers this conversation, so you can ask natural "
            "follow-up questions. You can type, or use the 🎙️ mic below to ask "
            "by voice — answers can be read aloud with 🔊 Listen."
        )

    with st.expander("🎙️ Voice", expanded=True):
        voice_transcript = voice.render_mic_input(key="sidebar_mic")
        if voice_transcript:
            st.session_state["pending_question"] = voice_transcript
            st.rerun()

        voice_output_enabled = st.checkbox(
            "🔊 Auto-read new answers aloud", value=False,
            help="A '🔊 Listen' button is always available under each answer. "
                 "Turn this on to have new answers read aloud automatically as "
                 "they arrive.",
        )
        st.session_state["voice_output_enabled"] = voice_output_enabled

    with st.expander("⚙️ Engine settings"):
        retrieval_backend = st.selectbox(
            "Retrieval backend",
            options=["tfidf", "faiss"],
            format_func=lambda x: {"tfidf": "TF-IDF (scikit-learn)", "faiss": "FAISS vector DB"}[x],
            help="FAISS uses dense embeddings + an approximate-nearest-neighbor "
                 "index instead of sparse TF-IDF cosine similarity.",
        )
        router_backend = st.selectbox(
            "Query routing",
            options=["rule_based", "llm"],
            format_func=lambda x: {"rule_based": "Rule-based (keyword + similarity)",
                                    "llm": "LLM-based classification"}[x],
            help="LLM-based routing asks the configured LLM backend to pick the "
                 "route directly. Automatically falls back to rule-based if no "
                 "LLM API key is configured.",
        )

    st.divider()
    st.page_link("pages/1_Admin_Dashboard.py", label="📊 Admin / Analytics Dashboard")

assistant = get_assistant(retrieval_backend, router_backend)

ui.render_header(
    config.ASSISTANT_NAME,
    f"AI Training Assistant for New Employees at {config.COMPANY_NAME} &nbsp;·&nbsp; "
    f"retrieval: <code>{assistant.retrieval_backend}</code> &nbsp;·&nbsp; "
    f"router: <code>{assistant.router_backend}</code>",
)

ui.render_llm_status(assistant.llm)
ui.render_web_search_status(assistant.web_search, assistant.llm)

for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])
        if msg["role"] == "assistant" and msg.get("web_sources"):
            with st.expander(f"🌐 Web sources ({len(msg['web_sources'])})"):
                for w in msg["web_sources"]:
                    st.markdown(f"- [{w['title']}]({w['link']})")
        elif msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"📎 Sources ({len(msg['citations'])})"):
                for c in msg["citations"]:
                    st.markdown(f"- `{c}`")
        if msg["role"] == "assistant" and msg.get("query_id"):
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
            with fb_col1:
                if st.button("👍", key=f"up_{i}"):
                    analytics.log_feedback(msg["query_id"], "up")
                    st.toast("Thanks for the feedback!")
            with fb_col2:
                if st.button("👎", key=f"down_{i}"):
                    analytics.log_feedback(msg["query_id"], "down")
                    st.toast("Thanks — noted for review.")
        if msg["role"] == "assistant":
            voice.render_listen_button(msg["content"], key=f"listen_{i}")

pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask a question about onboarding, your role, or policies...")
question = pending or user_input

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing and retrieving..."):
            result = assistant.ask(question, session_id=st.session_state["session_id"])

        st.markdown(result.answer)

        route_display = ROUTE_LABELS_DISPLAY.get(result.route, result.route)
        meta_lines = [f"**Route:** {route_display}  (confidence: {result.routing_confidence:.2f})"]
        if result.used_followup_memory:
            meta_lines.append("🧠 **Memory:** treated as a follow-up — used the previous question to pull more relevant detail.")
        meta = "\n\n".join(meta_lines)
        st.caption(meta)

        if result.web_sources:
            with st.expander(f"🌐 Web sources ({len(result.web_sources)})"):
                for w in result.web_sources:
                    st.markdown(f"- [{w['title']}]({w['link']})")
        elif result.citations:
            with st.expander(f"📎 Sources ({len(result.citations)})"):
                for rc in result.retrieved:
                    st.markdown(f"- `{rc.chunk.citation}` (similarity: {rc.score:.2f})")

        with st.expander("Debug: routing scores"):
            st.write(result.routing_reason)
            st.json(result.routing_scores)

        voice.render_listen_button(
            result.answer,
            key="listen_live",
            autoplay=st.session_state.get("voice_output_enabled", False),
        )

    st.session_state["messages"].append({
        "role": "assistant",
        "content": result.answer,
        "meta": meta,
        "citations": result.citations,
        "web_sources": result.web_sources,
        "query_id": result.query_id,
    })

    # Persist the full conversation so it shows up in "Recent chats" and can
    # be resumed later (even after a page refresh or app restart).
    first_user_msg = next(
        (m["content"] for m in st.session_state["messages"] if m["role"] == "user"),
        question,
    )
    analytics.save_conversation(
        st.session_state["session_id"], first_user_msg, st.session_state["messages"]
    )

    st.rerun()
