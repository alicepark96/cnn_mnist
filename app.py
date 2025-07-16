# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

st.title("✍️ 손글씨 숫자 인식기 (MNIST + CNN)")

model = load_model("cnn_mnist.h5")

uploaded_file = st.file_uploader("28x28 크기의 손글씨 이미지를 업로드하세요 (흑백)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # 흑백
    image = ImageOps.invert(image)  # 흰 배경, 검은 글씨
    image = image.resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.image(image.resize((140, 140)), caption="입력된 이미지", use_column_width=False)
    st.subheader(f"🧠 모델 예측 결과: {predicted_digit}")
