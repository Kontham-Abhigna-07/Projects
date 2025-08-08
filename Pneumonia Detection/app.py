import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


@st.cache(allow_output_mutation=True)
def get_model():
    return load_model('pneumonia_model.h5')

model = get_model()

IMG_DIM = 64  
st.header("Chest X-ray Pneumonia Classifier")

uploaded_img = st.file_uploader("Choose an X-ray image file", type=['png', 'jpg', 'jpeg'])

if uploaded_img:
    img = Image.open(uploaded_img).convert('RGB').resize((IMG_DIM, IMG_DIM))
    
    # ✅ Fixed deprecated argument
    st.image(img, caption='Uploaded X-ray', use_container_width=True)

    img_data = np.asarray(img) / 255.0
    img_data = img_data[np.newaxis, ...]

    with st.spinner("Analyzing X-ray..."):
        pred_prob = model.predict(img_data)[0][0]

    confidence = float(pred_prob)
    confidence = min(max(confidence, 0.0), 1.0)

    if confidence > 0.5:
        diagnosis = "🟥 PNEUMONIA"
        confidence = confidence
    else:
        diagnosis = "🟩 Normal"
        confidence = 1 - confidence

    st.markdown(f"### 🧠 Prediction: **{diagnosis}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2%}**")
