import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import gdown

# Page Configuration
st.set_page_config(page_title="🧠 Brain Tumor Detector", layout="centered")

# CSS Styling
st.markdown("""
    <style>
    html, body, #root > div {
        height: 100%;
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
        overflow-x: hidden;
    }
    .main {
        min-height: 100vh;
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: transparent !important;
    }
    .main-card {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
        padding: 20px 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 64, 55, 0.5);
        color: white;
        margin-bottom: 40px;
    }
    section[data-testid="stFileUploader"] {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #fff;
    }
    section[data-testid="stFileUploader"] label {
        font-size: 28px !important;
        font-weight: 900 !important;
        color: #ffffff !important;
        text-shadow: 2px 2px 8px #000000;
        margin-bottom: 0.5rem;
    }
    section[data-testid="stFileUploader"] div[role="button"] {
        background-color: rgba(255, 255, 255, 0.95);
        color: #000 !important;
        font-weight: bold !important;
        border-radius: 10px;
        padding: 1rem;
    }
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        color: #fff;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: white;
        font-size: 14px;
        margin-top: 50px;
    }
    a {
        color: #ffffff;
        text-decoration: none;
        font-weight: 600;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with gdown from Google Drive
@st.cache_resource
def load_model():
    model_path = "models/model.h5"
    if not os.path.exists(model_path):
        st.info("⏬ Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1KoT_RkJo0cH-V5y54LaRdvdyh1AIOR6J"
  # Replace YOUR_FILE_ID
        gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Title
st.markdown("<h1 style='text-align:center;'>🧠 Brain Tumor Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Upload an MRI scan and press Predict to analyze tumor type</h4>", unsafe_allow_html=True)

# Main card
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.image(
        "https://lakezurichopenmri.com/wp-content/uploads/2025/04/MRI-for-Brain-Tumor-Detection-scaled.jpg",
        caption="MRI Brain Scan",
        use_container_width=True
    )

    uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"], label_visibility="visible")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
        st.image(image, caption="🖼️ Uploaded MRI", use_container_width=True)

        if st.button("🔍 Predict Tumor Now"):
            with st.spinner('Analyzing MRI scan...'):
                img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
                prediction = model.predict(img_array)
                classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                predicted_class = classes[np.argmax(prediction)]
                confidence = float(np.max(prediction))

                st.markdown(
                    f"<div class='prediction-box'>✅ Prediction: <b>{predicted_class}</b><br>🧪 Confidence: <b>{confidence:.2%}</b></div>",
                    unsafe_allow_html=True
                )
                st.progress(confidence)

                st.markdown("### 🔬 Class Probabilities")
                fig = go.Figure(go.Bar(
                    x=prediction[0],
                    y=classes,
                    orientation='h',
                    marker=dict(color=prediction[0], colorscale='Blues'),
                    text=[f"{p*100:.2f}%" for p in prediction[0]],
                    textposition="auto"
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(title='Confidence', range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Made with ❤️ by <a href='https://www.linkedin.com/in/aaqib-rashid-mir/' target='_blank'>Aaqib Rashid Mir</a> |
        <a href='https://github.com/mtechbro94' target='_blank'>GitHub</a>
    </div>
""", unsafe_allow_html=True)
