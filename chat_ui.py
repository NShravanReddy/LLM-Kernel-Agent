import streamlit as st
from answer_model2 import AAgent

st.set_page_config(page_title="🧠 Triton Kernel Assistant", layout="wide")
st.title("🤖 Triton Kernel Assistant")

# --- Sidebar for API Key and Adapter ---
st.sidebar.header("🔐 API Configuration")
api_key = st.sidebar.text_input("Enter your Cerebras API Key", type="password")
adapter_type = st.sidebar.selectbox("Adapter Type", [None, "sft", "grpo"])

# Initialize agent only if API key is present
if api_key:
    try:
        st.session_state.agent = AAgent(adapter_type=adapter_type, api_key=api_key)
    except Exception as e:
        st.error(f"❌ Error initializing agent: {e}")
        st.stop()
else:
    st.warning("Please enter a valid API key in the sidebar to begin.")
    st.stop()

# --- Chat Interface ---
st.subheader("💬 Enter your message(s)")

input_mode = st.radio("Input Mode", ["Single Message", "Batch Messages"], horizontal=True)

if input_mode == "Single Message":
    user_input = st.text_area("Type your message here:", height=150)
    if st.button("Send"):
        if user_input.strip():
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.agent.cerebras_chat_completion([
                        {"role": "user", "content": user_input.strip()}
                    ])
                    st.success("✅ Response")
                    st.write(response)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
else:
    batch_input = st.text_area("Enter multiple prompts (one per line):", height=200)
    if st.button("Send Batch"):
        prompts = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]
        if prompts:
            with st.spinner("Generating batch responses..."):
                try:
                    responses = st.session_state.agent.cerebras_chat_completion(prompts)
                    st.success("✅ Responses")
                    for i, (prompt, resp) in enumerate(zip(prompts, responses), start=1):
                        st.markdown(f"**Prompt {i}:** `{prompt}`")
                        st.write(resp)
                        st.markdown("---")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
