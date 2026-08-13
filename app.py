import os
import joblib
import streamlit as st
from src.preprocessing import clean_text

MODEL_PATH = "models/spam_detector.joblib"

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message and the machine-learning model will classify it as Spam or Not Spam.")

if not os.path.exists(MODEL_PATH):
    st.warning("The trained model is not available yet.")
    st.code("python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free prize..."
)

if st.button("Check Message"):
    if not message.strip():
        st.info("Please enter a message.")
    else:
        cleaned = clean_text(message)
        prediction = model.predict([cleaned])[0]

        if prediction == 1:
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: NOT SPAM")
