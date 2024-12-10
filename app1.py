import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import speech_recognition as sre
import os
import json
import time

model = load_model('improved_model.h5')

emotion_map = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Angry',
    4: 'Fearful',
    5: 'Disgust',
    6: 'Surprised'
}

def plot_waveform(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    st.pyplot(plt)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time", sr=sr)
    plt.colorbar(format="%+2.0f dB")
    plt.title("MFCC")
    plt.tight_layout()
    st.pyplot(plt)
    mfcc = np.mean(mfcc, axis=1)
    return mfcc

def transcribe_audio(file_path):
    recognizer = sre.Recognizer()
    with sre.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sre.UnknownValueError:
        return "Speech not recognized"
    except sre.RequestError:
        return "Service unavailable"

def record_audio():
    recognizer = sre.Recognizer()
    with sre.Microphone() as source:
        st.markdown('<div class="custom-listening">Listening... Please speak.</div>', unsafe_allow_html=True)
        audio_data = recognizer.listen(source, timeout=5)
        file_name = "recorded_audio.wav"
        with open(file_name, "wb") as f:
            f.write(audio_data.get_wav_data())
        return file_name
    
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.samplerate = 16000

    def recv(self, frame: av.AudioFrame):
        audio = np.frombuffer(frame.to_ndarray(), np.int16).astype(np.float32)
        audio = librosa.resample(audio, orig_sr=frame.sample_rate, target_sr=self.samplerate)
        return audio

st.markdown("""
    <style>
        /* Add Bootstrap CSS */
        @import url('https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css');

        /* Set background image */
        .stApp {
        background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
        background-size: cover;
        background-attachment: fixed;
        color: #e6e6ff; /* Default text color */
    }
    .custom-listening {
            color: black; 
            background-color: #f0ff00; 
            padding: 10px; 
            border-radius: 5px; 
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
   
        /* Customize Streamlit header */
        .css-1v0mbdj {
            text-align: center;
            color: white !important;
        }

        /* Customize file uploader button */
        .css-1p4hg4l {
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
        }

        .css-1p4hg4l:hover {
            background-color: rgba(255, 255, 255, 0.7);
            color: black;
        }

        .stButton>button {
            background-color: #1a73e8;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }

        .stButton>button:hover {
            background-color: #0c63e4;
        }
        .result {
        background-color: #54ca80; /* Radium Bright Yellow */
        color: #0f2027; /* Dark text for contrast */
        font-size: 1.8rem; /* Larger text for emphasis */
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .score{
        background-color: #f79834; /* Radium Bright Yellow */
        color: #0f2027; /* Dark text for contrast */
        font-size: 1.8rem; /* Larger text for emphasis */
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .Waveform {
        text-align: center;
        color: #ffcc00; /* Bright yellow color */
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom:20px;
    }

    /* Customize MFCC Features title */
    .extract_featuress {
        text-align: center;
        color: #ff80ff; /* Bright pink color */
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 10px;
        
    }
    .custom-title{
        text-align: center;
        color: white; /* Change color to white */
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .transcription {
            font-size: 20px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 10px;
            text-align: center;
            margin-top : 10px;
        }
        .text-result {
            font-size: 16px;
            color: #000000; /* black for emphasis */
            background-color: #F9F9F9; /* Light grey background */
            padding: 10px;
            border: 1px solid #DDDDDD; /* Light grey border */
            border-radius: 5px;
            line-height: 1.5; /* Increase line height for readability */
            margin-top: 5px;
            margin-bottom: 5 px;
        }

    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-title">Speech Emotion Detection 🗣️</div>', unsafe_allow_html=True)

option = st.radio("Choose Input Method", options=["Upload Audio File", "Record via Microphone"])
file_path = None

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3", "opus"])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_path = "temp_audio"
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        if uploaded_file.name.endswith(".opus"):
            wav_file_path = "temp_audio.wav"
            audio, sr = librosa.load(file_path, sr=None)
            sf.write(wav_file_path, audio, sr)
            file_path = wav_file_path

elif option == "Record via Microphone":
    if st.button("🎤 Start Recording"):
        file_path = record_audio()
        
#initializing before.
predicted_label = None
predicted_probability = None

if file_path:
    st.audio(file_path)
    st.markdown('<div class="transcription">Transcription of Audio:</div>', unsafe_allow_html=True)
    transcription = transcribe_audio(file_path)
    st.markdown(f"<div class='text-result'>{transcription}</div>", unsafe_allow_html=True)

    # st.audio(file_path)
    plot_waveform(file_path)
    st.markdown('<div class="Waveform">Waveform of Audio</div>', unsafe_allow_html=True)

    features = extract_features(file_path)
    st.markdown('<div class="extract_featuress">MFCC Features</div>', unsafe_allow_html=True)
    features = features.reshape(1, 1, features.shape[0])
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_probability = np.max(prediction)
    predicted_label = emotion_map.get(predicted_index, "Unknown")  # Ensures `predicted_label` is defined
    st.markdown(f'<div class="result">Predicted Emotion: {predicted_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="score">Confidence Score: {predicted_probability}</div>', unsafe_allow_html=True)


## User feedback code
feedback_folder = "User_feedback"

if not os.path.exists(feedback_folder):
    os.makedirs(feedback_folder)

st.markdown("""
    <style>
        /* Make text bold and add padding */
        .stTextInput>label {
            color: white;
            font-weight: bold;
            font-size: 16px; /* Optional: Set font size */
        }
        .white-text {
            color: white;
            background-color: transparent;
            font-weight: bold;  /* Bold the text */
            padding: 10px 20px; /* Add padding to the text */
            margin-top: 20px;   /* Add top margin */
        }
        
        /* Style for the feedback radio button labels */
        .stRadio>label {
            color: white;
            font-weight: bold;  /* Make radio button labels bold */
            margin-top: 10px;   /* Space above the radio buttons */
            font-size: 16px;    /* Optional: set font size for labels */
        }

        /* Styling the input options for radio buttons */
        .stRadio input {
            color: white;
            background-color: transparent !important;
        }

        /* Styling the options text in the radio buttons */
        .stRadio div div p {
            color: white !important;
            font-weight: bold; /* Make the option text bold */
            margin-left: 10px;  /* Space between the radio button and the text */
        }

        /* Text area styling for feedback input */
        .stTextArea textarea {
            color: black !important;  /* Make text black inside the text area */
            background-color: white !important;  /* White background for the text area */
            padding: 10px;  /* Padding inside the text area */
            border-radius: 5px; /* Optional: rounded corners for text area */
            font-size: 14px;  /* Optional: change font size */
            width: 100%; /* Ensure the text area fills the container width */
        }

        /* Label for the text area */
        .stTextArea>label {
            color: white;
            font-weight: bold;  /* Make the label text bold */
            margin-top: 10px;  /* Add space between label and text area */
        }

        /* Styling the submit button */
        .stButton button {
            color: white;
            font-weight: bold;  /* Make button text bold */
            padding: 10px 20px;  /* Add padding around the button text */
            background-color: #4CAF50;  /* Green background for the button */
            border-radius: 5px; /* Rounded corners for the button */
            margin-top: 20px;  /* Add space above the button */
        }
        .stButton button:hover {
            background-color: #388E3C; /* Dark green background when hovered */
            border-color- #000000;
        }
        /* Additional styling for the feedback text area */
        .stTextArea {
            margin-top: 20px; /* Space above the text area */
        }

    </style>
""", unsafe_allow_html=True)

feedback = st.radio(
    "Was the predicted emotion correct?",
    options=["Yes", "No"],
    index=0,
    label_visibility="visible"
)
correct_emotion = None
additional_feedback = None

# If "No," allow the user to input the correct emotion

if feedback == "No":
    correct_emotion = st.text_input("Please specify the correct emotion:")
    additional_feedback = st.text_area("Additional feedback (optional):")
else:
    additional_feedback = st.text_area("Additional feedback (optional):")

# Save feedback on button click
if st.button("Submit Feedback"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"{feedback_folder}/feedback_{timestamp}.txt"

    # If feedback is "Yes", use predicted emotion, otherwise, use the user input (or "Not Provided" if empty)
    final_correct_emotion = predicted_label if feedback == "Yes" else correct_emotion or "Not Provided"
    
    # Default "N/A" for feedback if not provided
    user_feedback = additional_feedback if additional_feedback else "N/A"

    # Prepare the feedback text
    feedback_text = f"""
    Predicted Emotion: {predicted_label if predicted_label else "Not Available"}
    Confidence Score: {predicted_probability if predicted_probability is not None else 0.0:.2f}
    Correct Emotion: {final_correct_emotion}
    Feedback: {user_feedback}
    """




    # Save feedback to a file
    with open(file_name, "w") as feedback_file:
        feedback_file.write(feedback_text)

    # Display confirmation and submitted feedback
    st.text_area("Feedback Submitted", feedback_text, height=200)
    st.markdown('<div class="white-text">Thank you for your feedback! It has been saved.</div>', unsafe_allow_html=True)
    
    
    
st.markdown("---")  # Horizontal line for separation
st.markdown(
    """
    <style>
    .subheader {
        color: white; /* Change text color to white */
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    <div class="subheader">Glossary of Useful Terms</div>
    """,
    unsafe_allow_html=True,
)

# Glossary of terms as static content
st.markdown(
    """
    - **LSTM**: Long Short Term Memory (LSTM) is a type of recurrent neural network (RNN) designed to capture dependencies over time, making it suitable for sequential data. It addresses the problem of vanishing gradients in traditional RNNs.
    - **SVM**: Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It finds the best hyperplane that separates data points into distinct classes.
    - **Confidence Score**: A measure of how confident the model is about its prediction.
    - **SER (Speech Emotion Recognition)**: A technique used to identify emotions from speech signals.
    - **Toronto TESS Dataset**: A dataset used for analyzing emotions from speech, featuring audio clips with different emotions.
    - **MFCC (Mel-Frequency Cepstral Coefficients)**: Features extracted from audio signals to represent their characteristics for analysis.
    - **Waveform**: A visual representation of an audio signal's amplitude over time.
    - **Streamlit**: is an open-source Python framework designed to create interactive and data-driven web applications for machine learning and data science projects.
    - **.h5 Models**: is commonly used to store pre-trained machine learning models created with deep learning frameworks like TensorFlow or Keras. It is based on the HDF5 (Hierarchical Data Format) standard, which efficiently stores large amounts of structured data.
    - **Audio extensions**: audio files in formats like .wav, .opus, and .mp3, enabling seamless processing of diverse audio types for speech emotion recognition.
    - **Librosa**: A Python library for audio and music analysis, used for feature extraction like MFCCs.
    - **Speech_recognition**: A library for recognizing speech from audio, supporting various engines and APIs.
    - **Streamlit_webrtc**: A Streamlit component enabling real-time audio/video streaming and processing in web apps.
    - **SoundFile**:A library for reading and writing audio files in various formats with high fidelity.
    - **Webrtc_streamer**: A Streamlit_webrtc utility for handling WebRTC streams in real-time applications.
    - **WebRtcMode**: Defines modes for WebRTC streaming, such as sending, receiving, or both, in real-time.
    - **AudioProcessorBase**: A base class in Streamlit_webrtc for creating custom audio processing pipelines.
    - **Jupyter Notebook**: Created Model using Jupyter Notebook.
    
    
    
    """
)
