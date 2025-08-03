import streamlit as st
from answer_model2 import AAgent

st.set_page_config(page_title="🤖 Triton Kernel Assistant", layout="centered")

st.title("🤖 Triton Kernel Assistant")
st.write("Generate optimized Triton kernel code for your task. Provide your API key and prompt below.")

# Input field for API key
api_key = st.text_input("🔑 Enter API Key", type="password")

# Adapter type dropdown (you can modify these)
adapter_type = st.selectbox("🧠 Select Model Type", ["cerebras", "openai", "mistral"])

# Text input area for prompt
prompt = st.text_area("💬 Prompt for Kernel Code Generation", height=150)

# Generate button
if st.button("🚀 Generate Code"):
    if not api_key:
        st.error("Please provide your API key.")
    elif not prompt:
        st.error("Please enter a prompt.")
    else:
        try:
            agent = AAgent(adapter_type=adapter_type, api_key=api_key)
            response = agent.generate(prompt)
            st.code(response, language="python")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
