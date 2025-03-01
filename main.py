import whisper
import streamlit as st
import librosa
import tempfile
import moviepy as mp
from io import BytesIO
import time


@st.cache_resource
def load_model():
    return whisper.load_model("large")


def create_srt_text(segments):
    srt_text = ""
    for i, segment in enumerate(segments, start=1):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        start_time = "{:02}:{:02}:{:06.3f}".format(
            int(start // 3600),
            int((start % 3600) // 60),
            start % 60
        ).replace(".", ",")

        end_time = "{:02}:{:02}:{:06.3f}".format(
            int(end // 3600),
            int((end % 3600) // 60),
            end % 60
        ).replace(".", ",")

        srt_text += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_text


if 'result' not in st.session_state:
    st.session_state.result = None

st.sidebar.title("🎥 Video to SRT Converter")
uploaded_file = st.sidebar.file_uploader(
    "Upload a video/audio file",
    type=["mp3", "wav", "mp4", "mov"],
    help="Supported formats: MP3, WAV, MP4, MOV"
)

st.title("Automatic Caption Generator")
st.caption("Powered by OpenAI Whisper AI")


def process_audio(file):
    try:
        # For video files
        if file.type.startswith('video/'):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_vid:
                tmp_vid.write(file.getvalue())
                tmp_vid.seek(0)

                with st.spinner("🎞️ Extracting audio from video..."):
                    video = mp.VideoFileClip(tmp_vid.name)
                    audio = video.audio

                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
                        audio.write_audiofile(tmp_audio.name, codec='pcm_s16le', fps=16000)
                        audio_array, sr = librosa.load(tmp_audio.name, sr=16000, mono=True)

                    audio.close()
                    video.close()
            return audio_array

        # For audio files
        else:
            audio_bytes = file.read()
            return librosa.load(BytesIO(audio_bytes), sr=16000, mono=True)[0]

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()


if uploaded_file:
    st.sidebar.success("✅ File uploaded successfully!")

    if st.session_state.result is None:
        with st.status("🔍 Processing...", expanded=True) as status:
            try:
                st.write("Loading and processing file...")
                audio_array = process_audio(uploaded_file)

                st.write("Initializing model...")
                model = load_model()

                st.write("Transcribing content...")
                start_time = time.time()
                st.session_state.result = model.transcribe(
                    audio_array,
                    task="transcribe",
                    verbose=True,
                    language="en"
                )

                processing_time = time.time() - start_time
                status.update(label=f"✅ Processing complete! ({processing_time:.2f}s)",
                              state="complete", expanded=False)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.stop()

    if st.session_state.result:
        st.subheader("Generated Captions")
        col1, col2 = st.columns([3, 1])

        with col1:
            caption_container = st.empty()
            full_captions = []

            for segment in st.session_state.result["segments"]:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()

                start_mm = f"{int(start // 60):02}:{start % 60:06.3f}"
                end_mm = f"{int(end // 60):02}:{end % 60:06.3f}"
                caption_line = f"[{start_mm} --> {end_mm}] {text}"

                full_captions.append(caption_line)
                display_text = "\n\n".join(full_captions)
                caption_container.markdown(f"```\n{display_text}\n```")
                time.sleep(0.2)

        with col2:
            srt_content = create_srt_text(st.session_state.result["segments"])
            st.download_button(
                label="📥 Download SRT",
                data=srt_content,
                file_name="captions.srt",
                mime="text/plain"
            )

        st.balloons()
else:
    st.info("👈 Please upload a file to get started")

st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)