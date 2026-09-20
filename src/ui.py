"""
Shared branding/UI helpers for the Streamlit app and the admin dashboard.

Keeping this in one place means both `app.py` and `pages/1_Admin_Dashboard.py`
render an identical header, color theme, and status banners.
"""
import base64
from pathlib import Path

import streamlit as st

from . import config

ASSETS_DIR = config.PROJECT_ROOT / "assets"

# ---------------------------------------------------------------------------
# Light / dark colour tokens.
# inject_css() prepends one of these blocks (as a :root{...} CSS var block)
# ahead of assets/theme.css, so the same stylesheet renders in either mode.
# ---------------------------------------------------------------------------
_THEME_TOKENS = {
    "light": {
        "--ob-indigo": "#4338CA",
        "--ob-indigo-dark": "#362FA0",
        "--ob-teal": "#0D9488",
        "--ob-slate-900": "#1B2130",
        "--ob-bg": "#F7F8FC",
        "--ob-header-bg": "#FFFFFF",
        "--ob-sidebar-bg": "#F4F6FB",
        "--ob-card-bg": "#FFFFFF",
        "--ob-chip-bg": "#EEF0FA",
        "--ob-text": "#1B2130",
        "--ob-text-secondary": "#5B6474",
        "--ob-border": "#E4E8F0",
        "--ob-radius": "16px",
        "--ob-green": "#16A34A",
        "--ob-amber": "#D97706",
        "--ob-red": "#DC2626",
        "--ob-status-live-bg": "#ECFDF5",
        "--ob-status-live-border": "#A7F3D0",
        "--ob-status-live-text": "#065F46",
        "--ob-status-offline-bg": "#EFF3FE",
        "--ob-status-offline-border": "#C7D2FE",
        "--ob-status-offline-text": "#3730A3",
        "--ob-status-fallback-bg": "#FFFBEB",
        "--ob-status-fallback-border": "#FDE68A",
        "--ob-status-fallback-text": "#92400E",
    },
    "dark": {
        "--ob-indigo": "#818CF8",
        "--ob-indigo-dark": "#6366F1",
        "--ob-teal": "#2DD4BF",
        "--ob-slate-900": "#E5E9F5",
        "--ob-bg": "#0F1220",
        "--ob-header-bg": "#171B2C",
        "--ob-sidebar-bg": "#12162480",
        "--ob-card-bg": "#171B2C",
        "--ob-chip-bg": "#232945",
        "--ob-text": "#E9ECF7",
        "--ob-text-secondary": "#9AA3BD",
        "--ob-border": "#2A3049",
        "--ob-radius": "16px",
        "--ob-green": "#4ADE80",
        "--ob-amber": "#FBBF24",
        "--ob-red": "#F87171",
        "--ob-status-live-bg": "#0B2E24",
        "--ob-status-live-border": "#14532D",
        "--ob-status-live-text": "#6EE7B7",
        "--ob-status-offline-bg": "#1E2340",
        "--ob-status-offline-border": "#312E81",
        "--ob-status-offline-text": "#A5B4FC",
        "--ob-status-fallback-bg": "#332705",
        "--ob-status-fallback-border": "#78350F",
        "--ob-status-fallback-text": "#FCD34D",
    },
}


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def inject_css(theme: str = "light") -> None:
    """Injects colour tokens for the chosen theme ('light' or 'dark') plus
    the shared theme.css stylesheet. Safe to call every rerun."""
    tokens = _THEME_TOKENS.get(theme, _THEME_TOKENS["light"])
    root_vars = "\n".join(f"  {k}: {v};" for k, v in tokens.items())
    css_path = ASSETS_DIR / "theme.css"
    theme_css = css_path.read_text() if css_path.exists() else ""
    st.markdown(
        f"<style>\n:root {{\n{root_vars}\n}}\n{theme_css}\n</style>",
        unsafe_allow_html=True,
    )


def render_theme_toggle(location=st.sidebar) -> str:
    """Renders a 🌙/☀️ dark-mode toggle and returns the active theme name
    ('light' or 'dark'), persisted in st.session_state['theme']."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    is_dark = st.session_state["theme"] == "dark"
    with location:
        new_value = st.toggle("🌙 Dark mode", value=is_dark, key="ob_theme_toggle")
    new_theme = "dark" if new_value else "light"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()
    return st.session_state["theme"]


def page_icon():
    """Returns a usable page_icon for st.set_page_config (falls back to an emoji)."""
    icon_path = ASSETS_DIR / "favicon_180.png"
    return str(icon_path) if icon_path.exists() else "🧭"


def render_sidebar_logo() -> None:
    """Small compass mark + wordmark pinned above the sidebar nav."""
    logo_path = ASSETS_DIR / "logo_mark_512.png"
    if logo_path.exists():
        st.logo(str(logo_path), size="large")


def render_header(title: str, subtitle: str) -> None:
    """Branded header used at the top of every page (logo mark + title block)."""
    logo_path = ASSETS_DIR / "logo_mark_512.png"
    logo_html = ""
    if logo_path.exists():
        logo_html = f'<img src="data:image/png;base64,{_b64(logo_path)}" width="46" height="46"/>'
    st.markdown(
        f"""
        <div class="ob-header">
            {logo_html}
            <div>
                <p class="ob-title">{title}</p>
                <p class="ob-subtitle">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_llm_status(llm) -> None:
    """
    Replaces the old plain st.info/st.warning blocks with a styled status
    card that clearly distinguishes three states:
      - live: a real LLM backend (groq) is answering
      - offline (intentional): LLM_BACKEND is explicitly "extractive"
      - fallback (something broke): auto-detected a key but the call failed,
        so the session silently downgraded to extractive mode
    """
    if llm.backend != "extractive":
        st.markdown(
            f"""
            <div class="ob-status ob-status--live">
                <div class="ob-status-icon">✅</div>
                <div>
                    <b>Live LLM answers enabled</b>
                    <span class="ob-pill ob-pill--live">{llm.backend}</span>
                    <div class="ob-status-detail">
                        Responses are generated by {llm.backend} and grounded in the
                        retrieved onboarding documents.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if config.LLM_BACKEND == "extractive":
        st.markdown(
            """
            <div class="ob-status ob-status--offline">
                <div class="ob-status-icon">🧾</div>
                <div>
                    <b>Offline extractive mode</b>
                    <span class="ob-pill ob-pill--offline">no API calls</span>
                    <div class="ob-status-detail">
                        This is intentional (LLM_BACKEND=extractive). Answers are built
                        directly from the retrieved knowledge-base passages, with
                        citations, and no external API calls are made.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif llm.fallback_reason:
        st.markdown(
            f"""
            <div class="ob-status ob-status--fallback">
                <div class="ob-status-icon">⚠️</div>
                <div>
                    <b>Falling back to offline mode</b>
                    <span class="ob-pill ob-pill--fallback">fallback active</span>
                    <div class="ob-status-detail">
                        {llm.fallback_reason}<br/>
                        Most likely cause: the API key is missing, invalid, revoked, or
                        out of credit. Fix it in <code>.env</code>
                        (<code>GROQ_API_KEY</code>),
                        then use "🔄 New conversation" or restart the app — the backend
                        is only re-checked when a fresh session/cached resource is created.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # else: no API key configured and no fallback triggered — the assistant
    # is simply running in offline extractive mode by default. No banner is
    # shown for this case to keep the chat UI clean and uncluttered.


def render_web_search_status(web_search, llm) -> None:
    """Small status line showing whether live web-search fallback (Serper)
    is configured. Only rendered as a compact caption, since it's secondary
    to the main LLM status banner above it."""
    if llm.groq_web_search_enabled:
        st.caption("🌐 Web search fallback: **enabled** via Groq browser search — used automatically "
                   "for questions that don't match the onboarding knowledge base.")
    elif web_search.enabled:
        st.caption("🌐 Web search fallback: **enabled** via Serper — used automatically "
                   "for questions that don't match the onboarding knowledge base.")
    else:
        st.caption("🌐 Web search fallback: **disabled**. Set `SERPER_API_KEY` in `.env` "
                   "to let the assistant search the live web for questions outside the "
                   "onboarding docs, instead of just declining them. Free key: "
                   "[serper.dev](https://serper.dev), or set `GROQ_API_KEY` to use Groq browser search.")
