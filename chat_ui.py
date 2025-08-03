import streamlit as st
from answer_model2 import AAgent  # Adjust this path if your file is in a folder

st.set_page_config(page_title="Chat with Triton LLM", layout="wide")
st.title("🤖 Chat with Triton Kernel Assistant")

# Initialize session state for chat history and agent
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of {"role": "user"/"assistant", "content": "..."}
if "agent" not in st.session_state:
    st.session_state.agent = AAgent(adapter_type=None)  # or "sft", "grpo"

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box at the bottom
prompt = st.chat_input("Type a prompt (e.g., 'Write a Triton kernel for softmax')...")

# On message submission
if prompt:
    # Show user message in UI
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.cerebras_chat_completion([
                {"role": "user", "content": prompt}
            ])
            st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
