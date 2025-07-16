# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(page_title="손글씨 숫자 인식기", layout="centered")
st.title("✍️ 직접 써보는 숫자 인식기 (MNIST + CNN)")

# 모델 불러오기
model = load_model("cnn_mnist.h5")

st.markdown("### 아래 칸에 숫자를 그려보세요!")
st.markdown("배경: 검정색, 선: 흰색 (마우스로 입력)")

# 캔버스 설정
canvas_result = st_canvas(
    fill_color="#000000",  # 캔버스 배경은 검정
    stroke_width=15,
    stroke_color="#FFFFFF",  # 흰색 펜
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 예측하기
if canvas_result.image_data is not None:
    # 1. 이미지 가져오기
    img = canvas_result.image_data[:, :, 0]  # 흑백으로 추출
    img = Image.fromarray(img)
    img = img.resize((28, 28))  # MNIST 크기로 리사이즈
    img = np.array(img).astype("float32") / 255.0
    img = 1 - img  # 배경은 흰색, 글씨는 검정으로 반전
    img = img.reshape(1, 28, 28, 1)

    # 2. 예측
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    st.markdown("### 🧠 모델이 예측한 숫자:")
    st.header(f"👉 {predicted_digit}")
