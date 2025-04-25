import streamlit as st
import requests
import os
from dotenv import load_dotenv


load_dotenv()
API_URL = os.getenv("API_URL")

if not API_URL:
    st.error("API_URL not found")

st.set_page_config(page_title="Audio Transcriber", layout="centered")
st.title("Audio Transcriber")

tab1, tab2 = st.tabs(["Upload Audio", "Transcribe YouTube"])

with tab1:
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader("Choose a file", type=["mp3"])

    if uploaded_file:
        st.audio(uploaded_file, format='audio/mp3')

    if uploaded_file and st.button("Transcribe"):
        with st.spinner("⏳ Transcribing..."):

            try:
                file_bytes = uploaded_file.getvalue()
                files = {"file": (uploaded_file.name, file_bytes)}

                response = requests.post(f"{API_URL}/transcribe", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Transcription successful!")
                    st.write("📝 **Transcription Result:**")
                    st.text(data["transcription"])
                    st.write(f"⏱️ Inference Time: `{data['inference_time']}s`")
                else:
                    st.error(f"❌ Failed: {response.status_code}")
                    st.code(response.text)

            except Exception as e:
                st.error("❌ An error occurred while sending the file to the API")
                st.exception(e)

with tab2:
    yt_link = st.text_input("Paste YouTube Link")

    if st.button("Transcribe from YouTube"):
        if yt_link:
            with st.spinner("Downloading and transcribing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/transcribe-youtube", json={"url": yt_link})

                    if response.status_code == 200:
                        data = response.json()
                        st.success("✅ Transcription successful!")
                        st.write("📝 **Transcription Result:**")
                        st.text(data["transcription"])
                        st.write(
                            f"⏱️ Inference Time: `{data['inference_time']}s`")
                    else:
                        st.error(f"❌ Failed: {response.status_code}")
                        st.code(response.text)

                except Exception as e:
                    st.error("❌ An error occurred while processing the request")
                    st.exception(e)
        else:
            st.warning("⚠️ Please enter a YouTube link first.")
