from . import config


class LLMClient:
    def __init__(self):
        self.backend = self._resolve_backend()
        self.fallback_reason = ""

    def _resolve_backend(self) -> str:
        if config.LLM_BACKEND in ("groq", "extractive"):
            return config.LLM_BACKEND
        # auto-detect
        if config.GROQ_API_KEY:
            return "groq"
        return "extractive"

    # ------------------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str, context_chunks=None) -> str:
        context_chunks = context_chunks or []
        extractive_fn = lambda: self._generate_extractive(user_prompt, context_chunks)
        return self._generate_with_fallback(system_prompt, user_prompt, extractive_fn)

    def generate_from_web(self, system_prompt: str, user_prompt: str, web_response) -> str:
        """Same LLM call path as `generate`, but falls back to a deterministic
        summary built from web search snippets (instead of doc chunks) if no
        LLM backend is available."""
        extractive_fn = lambda: self._generate_extractive_web(web_response)
        return self._generate_with_fallback(system_prompt, user_prompt, extractive_fn)

    def _generate_with_fallback(self, system_prompt: str, user_prompt: str, extractive_fn) -> str:
        if self.backend == "extractive":
            return extractive_fn()

        configured_backend = self.backend
        try:
            return self._generate_with_backend(configured_backend, system_prompt, user_prompt)
        except Exception as primary_error:
            alternate_backend = self._available_alternate_backend(configured_backend)
            if alternate_backend:
                try:
                    answer = self._generate_with_backend(alternate_backend, system_prompt, user_prompt)
                    self.backend = alternate_backend
                    self.fallback_reason = (
                        f"{configured_backend} request failed ({type(primary_error).__name__}); "
                        f"switched to {alternate_backend}."
                    )
                    return answer
                except Exception as alternate_error:
                    failure_reason = (
                        f"{configured_backend} request failed ({type(primary_error).__name__}); "
                        f"{alternate_backend} request also failed ({type(alternate_error).__name__})"
                    )
            else:
                failure_reason = f"{configured_backend} request failed ({type(primary_error).__name__})"

            # A bad key, exhausted credits, or a temporary provider outage should
            # never take down the local assistant. Switch this session to the
            # deterministic retrieval-backed mode instead.
            self.backend = "extractive"
            self.fallback_reason = f"{failure_reason}; using offline extractive mode."
            return extractive_fn()

    def _generate_with_backend(self, backend: str, system_prompt: str, user_prompt: str) -> str:
        if backend == "groq":
            return self._generate_groq(system_prompt, user_prompt)
        raise ValueError(f"Unsupported LLM backend: {backend}")

    @staticmethod
    def _available_alternate_backend(current_backend: str) -> str:
        # Only one live LLM backend (Groq) is configured in this build, so
        # there is no alternate provider to fail over to. A failed Groq call
        # falls straight through to offline extractive mode.
        return ""

    # ------------------------------------------------------------------
    def _generate_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Generate through Groq's OpenAI-compatible chat-completions API."""
        from openai import OpenAI

        client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    @property
    def groq_web_search_enabled(self) -> bool:
        return config.GROQ_WEB_SEARCH_ENABLED

    def generate_from_groq_web_search(self, system_prompt: str, user_prompt: str) -> str:
        """Answer an out-of-scope question with Groq's server-side browser search.

        Groq currently limits this tool to its GPT-OSS models, so a dedicated
        configurable model is used instead of the regular GROQ_MODEL.
        """
        if not self.groq_web_search_enabled:
            raise RuntimeError("Groq browser search is not configured.")

        from openai import OpenAI

        client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        resp = client.chat.completions.create(
            model=config.GROQ_WEB_SEARCH_MODEL,
            max_tokens=900,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tool_choice="required",
            tools=[{"type": "browser_search"}],
        )
        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("Groq browser search returned an empty answer.")
        return answer

    # ------------------------------------------------------------------
    def _generate_extractive(self, user_prompt: str, context_chunks) -> str:
        """
        Deterministic, no-API fallback. Produces a grounded answer by
        lightly summarizing/quoting the top retrieved chunks and always
        attaches citations. Used automatically when no API key is set,
        and by unit tests so the pipeline never depends on network access.
        """
        if not context_chunks:
            return (
                "I don't have a documented answer for that in the onboarding "
                "knowledge base. This looks like something to confirm with your "
                "manager or HR directly, since it may involve personal data, an "
                "approval, or a topic outside the current onboarding materials."
            )

        lines = ["Here's what the onboarding documents say:"]
        for rc in context_chunks:
            c = rc.chunk
            snippet = " ".join(c.text.split())
            if len(snippet) > 320:
                snippet = snippet[:317].rstrip() + "..."
            lines.append(f"\n• [{c.section or c.doc_title}] {snippet}")
        lines.append(
            "\n\nIf this doesn't fully answer your question, please check with "
            "your manager or the relevant team (HR/IT) for specifics."
        )
        return "\n".join(lines)

    @staticmethod
    def _generate_extractive_web(web_response) -> str:
        """Deterministic, no-API summary built from Serper web results.
        Used when web search succeeded but no LLM backend is available to
        write a synthesized answer."""
        if web_response is None or not web_response.ok:
            return (
                "I don't have a documented answer for that in the onboarding "
                "knowledge base, and I wasn't able to find anything useful on "
                "the web either. This is probably one to confirm with your "
                "manager or HR directly."
            )

        lines = ["I couldn't find this in our onboarding docs, but here's what I found online:"]
        if web_response.answer_box:
            lines.append(f"\n{web_response.answer_box}")
        for i, r in enumerate(web_response.results, start=1):
            snippet = " ".join(r.snippet.split())
            if len(snippet) > 240:
                snippet = snippet[:237].rstrip() + "..."
            lines.append(f"\n[{i}] **{r.title}** — {snippet}\n{r.link}")
        lines.append(
            "\n\nThis is general web information, not an internal company policy — "
            "please confirm anything important with your manager or HR."
        )
        return "\n".join(lines)
