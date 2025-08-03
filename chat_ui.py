import streamlit as st
from answer_model2 import AAgent

st.set_page_config(page_title="Chat with Triton LLM", layout="wide")
st.title("🤖 Triton Kernel Assistant")

# --- Step 1: Input API key securely ---
with st.sidebar:
    st.header("🔐 Cerebras API Key")
    api_key = st.text_input("Enter Cerebras API Key:", type="password")
    adapter_type = st.selectbox("Adapter Type", [None, "sft", "grpo"], index=0)

# --- Step 2: Save API key and initialize agent ---
if "agent" not in st.session_state or st.session_state.get("adapter_type") != adapter_type or st.session_state.get("api_key") != api_key:
    st.session_state.api_key = api_key
    st.session_state.adapter_type = adapter_type
    st.session_state.agent = AAgent(adapter_type=adapter_type, api_key=api_key)

# --- Step 3: Chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Step 4: Show previous chat ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Step 5: Chat input ---
prompt = st.chat_input("Ask about Triton kernel...")

if prompt:
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.cerebras_chat_completion([
                {"role": "user", "content": prompt}
            ])
            st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
