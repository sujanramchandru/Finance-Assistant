import streamlit as st
import requests
from streamlit_mic_recorder import mic_recorder
import tempfile
import time
import os

st.set_page_config(page_title="Financial Assistant", layout="centered")
st.title("📊 Financial Assistant")

# API URLs
ORCHESTRATOR_URL = "http://127.0.0.1:8000/receive_transcription"
TTS_URL = "http://127.0.0.1:8006/speak"  # Assuming your TTS agent runs on 8006

# Input options
input_mode = st.radio("Choose input mode:", ["Text", "Voice 🎙️"])

if input_mode == "Text":
    user_query = st.text_input("Enter your query:")
elif input_mode == "Voice 🎙️":
    audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop", key="mic")
    user_query = audio.get("text") if audio else ""

# Submit query
if user_query:
    st.markdown("### 🔍 Query:")
    st.write(user_query)

    with st.spinner("Processing your query..."):
        try:
            # 1. Send query to orchestrator
            orchestrator_resp = requests.post(ORCHESTRATOR_URL, json={"transcription": user_query}, timeout=120)
            result = orchestrator_resp.json()
            llm_text = result.get("llm_response", {}).get("response", "No response from LLM agent.")

            # 2. Display LLM Response
            st.markdown("### 🤖 Assistant's Answer:")
            st.success(llm_text)

            # 3. Send to TTS
            tts_resp = requests.post(TTS_URL, json={"text": llm_text})
            tts_msg = tts_resp.json().get("message", "")
            st.write("🔈 " + tts_msg)

        except requests.exceptions.Timeout:
            st.error("⏳ The request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
