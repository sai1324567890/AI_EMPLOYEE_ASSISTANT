"""
Admin / Analytics Dashboard.

Streamlit auto-discovers this as a second page (see the sidebar nav) for
any app started with `streamlit run app.py`. Reads from the same SQLite
analytics store that `app.py` writes to on every question asked.

Protected by a simple username/password gate (default admin / admin,
override via ADMIN_USERNAME / ADMIN_PASSWORD in .env).

Run with:
    streamlit run app.py
    (then use the sidebar nav, or open the "Admin Dashboard" page link)
"""
import pandas as pd
import streamlit as st

from src import analytics, config, ui

st.set_page_config(
    page_title=f"{config.ASSISTANT_NAME} · Admin Dashboard",
    page_icon=ui.page_icon(),
    layout="wide",
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
ui.inject_css(st.session_state["theme"])
ui.render_sidebar_logo()

with st.sidebar:
    is_dark = st.toggle(
        "🌙 Dark mode", value=(st.session_state["theme"] == "dark"), key="ob_theme_toggle_admin"
    )
    new_theme = "dark" if is_dark else "light"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()
    st.divider()
    st.page_link("app.py", label="⬅ Back to chat")


def _require_login() -> bool:
    """Renders a login card and gates the rest of the page until the admin
    signs in with ADMIN_USERNAME / ADMIN_PASSWORD (default admin / admin)."""
    if st.session_state.get("admin_authed"):
        return True

    st.markdown(
        """
        <div class="ob-login-wrap">
            <div class="ob-login-icon">🔐</div>
            <h3>Admin sign in</h3>
            <p>Enter your admin credentials to view analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, form_col, _ = st.columns([1, 1.4, 1])
    with form_col:
        with st.form("admin_login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="admin")
            submitted = st.form_submit_button("Log in", use_container_width=True, type="primary")

        if submitted:
            if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
                st.session_state["admin_authed"] = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
    return False


if not _require_login():
    st.stop()

with st.sidebar:
    if st.button("🚪 Log out", use_container_width=True):
        st.session_state["admin_authed"] = False
        st.rerun()

ui.render_header(
    "Admin / Analytics Dashboard",
    f"{config.ASSISTANT_NAME} — usage analytics and query insights",
)

stats = analytics.get_stats(top_n=15)

if stats.total_queries == 0:
    st.info(
        "No queries logged yet. Go to the main chat page and ask a few questions — "
        "this dashboard updates live from the same analytics store."
    )
    st.page_link("app.py", label="⬅ Back to chat")
    st.stop()

# --- Top-line metrics -------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total questions asked", stats.total_queries)
col2.metric("Sessions", stats.total_sessions)
total_feedback = stats.feedback_up + stats.feedback_down
satisfaction = f"{stats.feedback_up / total_feedback:.0%}" if total_feedback else "n/a"
col3.metric("👍 / 👎 feedback", f"{stats.feedback_up} / {stats.feedback_down}")
col4.metric("Satisfaction rate", satisfaction)

st.divider()

# --- Query category statistics ----------------------------------------------
st.subheader("Query Category Statistics")
route_df = pd.DataFrame(
    [{"route": r or "unknown", "count": c} for r, c in stats.route_distribution.items()]
).sort_values("count", ascending=False)

chart_col, table_col = st.columns([2, 1])
with chart_col:
    st.bar_chart(route_df.set_index("route"), color="#4338CA")
with table_col:
    st.dataframe(route_df, use_container_width=True, hide_index=True)

st.divider()

# --- User activity ------------------------------------------------------------
st.subheader("User Activity")
if stats.activity_by_day:
    activity_df = pd.DataFrame(stats.activity_by_day, columns=["day", "queries"])
    st.line_chart(activity_df.set_index("day"), color="#0D9488")
else:
    st.write("Not enough data yet for a daily activity trend.")

with st.expander("Recent query log"):
    recent = analytics.get_recent_queries(limit=100)
    if recent:
        recent_df = pd.DataFrame(
            recent,
            columns=["timestamp", "session_id", "question", "route", "confidence",
                     "llm_backend", "retrieval_backend", "citations"],
        )
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.write("No queries logged yet.")

st.divider()

# --- Recent conversations ----------------------------------------------------
st.subheader("Recent Conversations")
recent_convos = analytics.get_recent_conversations(limit=25)
if recent_convos:
    convo_df = pd.DataFrame(recent_convos, columns=["session_id", "title", "updated_at"])
    st.dataframe(convo_df, use_container_width=True, hide_index=True)
else:
    st.write("No saved conversations yet.")

st.divider()

# --- Frequently Asked Questions ----------------------------------------------
st.subheader("Frequently Asked Questions")
if stats.top_questions:
    faq_df = pd.DataFrame(stats.top_questions, columns=["question", "times_asked"])
    st.dataframe(faq_df, use_container_width=True, hide_index=True)
else:
    st.write("No repeated questions yet.")

st.divider()

# --- Feedback results ------------------------------------------------------------
st.subheader("User Feedback Results")
feedback_summary = analytics.get_feedback_summary()
if feedback_summary is None:
    st.write(
        "No feedback recorded yet. Feedback (👍 / 👎) is collected from the buttons "
        "under each answer on the chat page."
    )
else:
    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric("Total rated answers", feedback_summary["total"])
    fcol2.metric("Positive", feedback_summary["up"])
    fcol3.metric("Negative", feedback_summary["down"])
    st.progress(feedback_summary["satisfaction_rate"],
                text=f"Satisfaction rate: {feedback_summary['satisfaction_rate']:.1%}")
    if feedback_summary["comments"]:
        st.write("**Recent comments:**")
        comments_df = pd.DataFrame(
            feedback_summary["comments"], columns=["question", "rating", "comment", "timestamp"]
        )
        st.dataframe(comments_df, use_container_width=True, hide_index=True)

st.divider()
st.page_link("app.py", label="⬅ Back to chat")
