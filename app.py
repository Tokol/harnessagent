from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ai_agent import OperationalAgent, configured_api_key
from harness import ActionExecutor, AuthManager, JsonStore


def requests_password_disable(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    mentions_password = "password" in normalized or "passcode" in normalized
    disable_language = any(phrase in normalized for phrase in (
        "never ask", "do not ask", "don't ask", "stop asking", "remove password",
        "disable password", "turn off password", "no password", "unlock permanently",
    ))
    return mentions_password and disable_language


load_dotenv()
st.set_page_config(page_title="Harness Agent", page_icon="◈", layout="wide")

store = JsonStore(Path(__file__).parent / "agent_data")
auth = AuthManager(store)
auth_version = store.read("auth").get("updatedAt")

if "messages" not in st.session_state:
    saved = store.recent_history()
    st.session_state.messages = [{"role": row["role"], "content": row["content"]} for row in saved]
    if not st.session_state.messages:
        st.session_state.messages = [{"role": "assistant", "content": "Hey! How can I help?"}]
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = not auth.enabled()
if st.session_state.get("auth_version") != auth_version:
    st.session_state.auth_version = auth_version
    st.session_state.authenticated = not auth.enabled()
    saved = store.recent_history()
    st.session_state.messages = [{"role": row["role"], "content": row["content"]} for row in saved]
    if not st.session_state.messages:
        st.session_state.messages = [{"role": "assistant", "content": "Hey! How can I help?"}]

api_key = configured_api_key()
model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

st.title("Chat")
if not api_key:
    st.warning("Add `OPENAI_API_KEY` to `.env` and restart the app to connect.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if auth.enabled() and not st.session_state.authenticated:
    if not st.session_state.get("password_prompted"):
        st.session_state.messages.append({"role": "assistant", "content": "Please enter the password."})
        st.session_state.password_prompted = True
        st.rerun()
    if password := st.chat_input("Password…", key="password_input"):
        st.session_state.messages.append({"role": "user", "content": "••••••••"})
        if auth.verify(password):
            st.session_state.authenticated = True
            st.session_state.password_prompted = False
            st.session_state.messages.append({"role": "assistant", "content": "Correct. How can I help?"})
            st.rerun()
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Wrong password."})
            st.rerun()
    st.stop()

if prompt := st.chat_input("Message…"):
    was_auth_enabled = auth.enabled()
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    store.append_history("user", prompt)
    if not api_key:
        result = {"message": "I’m not connected yet. Add the API key to `.env`, then restart me.", "trace": []}
    elif requests_password_disable(prompt):
        try:
            result = ActionExecutor(store).execute("security_disable", {})
        except Exception as exc:
            result = {"message": f"I ran into an error: `{exc}`", "trace": [{"step": "Run failed", "detail": str(exc)}]}
    else:
        try:
            with st.spinner("Thinking…"):
                agent = OperationalAgent(store, api_key, model)
                result = agent.run(st.session_state.messages[-20:])
        except Exception as exc:
            result = {"message": f"I ran into an error: `{exc}`", "trace": [{"step": "Run failed", "detail": str(exc)}]}
    st.session_state.messages.append({"role": "assistant", "content": result["message"]})
    st.session_state.messages = st.session_state.messages[-20:]
    store.append_history("assistant", result["message"])
    st.session_state.last_trace = result["trace"]
    if not was_auth_enabled and auth.enabled():
        st.session_state.authenticated = False
    st.rerun()
