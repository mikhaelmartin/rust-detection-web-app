import streamlit as st
from PIL import Image
import numpy as np
import cv2
from model import YOLO

model = YOLO("weights.onnx")

st.set_page_config(layout='wide')

with st.container(border=False):
    st.title("Perumda Tirtawening Corrosion Analyzer")
    st.header("Created By : Ahmad Syaifullah")

col1, col2 = st.columns(2)
col1.subheader("Input Image")

uploaded_file = None
with col1.container(height=300,border=True):
    space = st.empty()
    space.container(height=50,border=False)
    uploader = st.empty()
    uploaded_file = uploader.file_uploader(
        "Choose a picture file", accept_multiple_files=False,
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        # img = img.resize((int(img.width*300.0/img.height),300))
        # st.write(img.height)
        st.image(img)
        space.empty()
        # space.container(height=int((300-img.height)/2))
        uploader.empty()


col2.subheader("Image Analysis")

with col2.container(height=300,border=True):
    if uploaded_file:
        img = Image.open(uploaded_file)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
        # Inference
        boxes, segments, _ = model(img, conf_threshold=0.5, iou_threshold=0.5)

        # Draw bboxes and polygons
        output_img = None
        if len(boxes) > 0:
            img = model.draw_and_visualize(img, boxes, segments, vis=False, save=True)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        st.image(img)
    


