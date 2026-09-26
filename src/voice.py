import json
import re

import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_mic_recorder import speech_to_text as _speech_to_text
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Voice input
# ---------------------------------------------------------------------------
def render_mic_input(key: str = "voice_input", language: str = "en"):
    """
    Renders a record/stop mic button. Returns a NEW transcript string the
    first time speech is recognized, otherwise None (so callers can safely
    do `if transcript: ...` without re-submitting the same recording on
    every rerun).

    Note: intentionally uses just_once=False and does its own "is this a
    new transcript" tracking in session_state, rather than the library's
    built-in just_once=True — that path re-returns a stale cached value
    (and can error) once a rerun happens without a fresh recording, e.g.
    right after we trigger st.rerun() to submit the transcribed question.

    Voice input relies on the browser's native SpeechRecognition API, which
    only Chromium-based browsers (Chrome, Edge, Chrome for Android)
    implement — it does NOT work in Safari or Firefox at all, and needs an
    HTTPS or localhost origin plus microphone permission to be granted.
    That's the most common cause of "voice doesn't work" — wrong browser,
    not a code issue. The component call is also wrapped in a try/except,
    since a third-party JS component erroring should surface a clear
    message rather than silently doing nothing or breaking the page.
    """
    if not MIC_AVAILABLE:
        st.caption(
            "🎙️ Voice input needs one more package: run "
            "`pip install streamlit-mic-recorder` (already in requirements.txt) "
            "and restart the app."
        )
        return None

    st.caption(
        "Works in **Chrome or Edge** (desktop or Android) — Safari and "
        "Firefox don't support browser speech recognition. First use will "
        "prompt for microphone permission."
    )

    try:
        raw = _speech_to_text(
            start_prompt="🎙️ Start speaking",
            stop_prompt="⏹️ Stop && use this",
            just_once=False,
            use_container_width=True,
            language=language,
            key=key,
        )
    except Exception as e:
        st.caption(
            f"⚠️ Voice input hit an error ({type(e).__name__}) — you can "
            "still type your question below."
        )
        return None

    last_key = f"_{key}_last_transcript"
    if raw and raw.strip() and raw.strip() != st.session_state.get(last_key):
        st.session_state[last_key] = raw.strip()
        return raw.strip()
    return None


# ---------------------------------------------------------------------------
# Voice output
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(r"\[[^\]]*\.(md|pdf|docx|txt)[^\]]*\]", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_MD_EMPHASIS_RE = re.compile(r"(\*\*|\*|__|_|`)")
_NUMBERED_CITE_RE = re.compile(r"\[\d+\]")


def _clean_for_speech(text: str) -> str:
    """Strip markdown/citation noise so the browser doesn't read out
    '[onboarding_faq.md#Section]' or literal asterisks out loud."""
    t = _MD_LINK_RE.sub(r"\1", text)
    t = _CITATION_RE.sub("", t)
    t = _NUMBERED_CITE_RE.sub("", t)
    t = _MD_EMPHASIS_RE.sub("", t)
    t = t.replace("•", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def render_listen_button(text: str, key: str, autoplay: bool = False) -> None:
    """Small '🔊 Listen' / '⏹ Stop' control that reads `text` aloud using
    the browser's native speechSynthesis API. Renders in its own isolated
    iframe, so it needs its own inline styling (page CSS doesn't reach in).

    autoplay=True speaks immediately on render (used for "auto-read new
    answers"). Browsers may still block this if the tab hasn't had any
    user interaction yet — clicking "🔊 Listen" always works regardless."""
    safe_text = json.dumps(_clean_for_speech(text))
    autoplay_js = f"speakBtn_{key}.click();" if autoplay else ""
    html = f"""
    <div style="display:flex; gap:8px; align-items:center;
                font-family: -apple-system, 'Segoe UI', sans-serif;">
      <button id="speak-{key}" style="
          display:flex; align-items:center; gap:6px;
          background:#EEF0FF; color:#4338CA; border:1px solid #C7D2FE;
          border-radius:8px; padding:5px 12px; font-size:13px; font-weight:600;
          cursor:pointer;">🔊 Listen</button>
      <button id="stop-{key}" style="
          display:flex; align-items:center; gap:6px;
          background:#F4F6FB; color:#5B6474; border:1px solid #E4E8F0;
          border-radius:8px; padding:5px 12px; font-size:13px; font-weight:600;
          cursor:pointer;">⏹ Stop</button>
    </div>
    <script>
      const ttsText_{key} = {safe_text};
      const speakBtn_{key} = document.getElementById('speak-{key}');
      const stopBtn_{key} = document.getElementById('stop-{key}');
      speakBtn_{key}.onmouseover = () => speakBtn_{key}.style.background = '#E0E4FF';
      speakBtn_{key}.onmouseout = () => speakBtn_{key}.style.background = '#EEF0FF';
      speakBtn_{key}.onclick = function() {{
          if (!('speechSynthesis' in window)) {{
              alert('Voice output is not supported in this browser.');
              return;
          }}
          window.speechSynthesis.cancel();
          const utter = new SpeechSynthesisUtterance(ttsText_{key});
          utter.rate = 1.0;
          utter.pitch = 1.0;
          window.speechSynthesis.speak(utter);
      }};
      stopBtn_{key}.onclick = function() {{
          window.speechSynthesis.cancel();
      }};
      {autoplay_js}
    </script>
    """
    components.html(html, height=42)
