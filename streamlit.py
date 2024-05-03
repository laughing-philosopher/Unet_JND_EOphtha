import streamlit as st
from PIL import Image
from processing import processing
import numpy as np
import cv2


def main():

    st.title("Unet_JND_EOphtha")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # print(uploaded_image)
        image = Image.open(uploaded_image)
        image_cv2 = np.array(image.convert('RGB'))
        st.image(image_cv2)
        output_image = processing(image_cv2)
        st.image(output_image)
        

if __name__ == "__main__":
    main()