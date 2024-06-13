import streamlit as st
from PIL import Image
from processing import processing
import numpy as np
import cv2


def main():
    image = Image.open('iitbbs logo.png')
    st.image(image, width=256)
    st.title("Image and Video Processing Lab")
    st.title("Unet_JND_EOphtha")

    threshold = st.number_input('Enter the threshold value between 0.5 and 0.99 (recommended value = 0.9)', min_value=0.5, max_value=0.99)
    batch_size = st.number_input('Enter Batch Size (larger batch size will be slower but more stable, recommended value = 4)', min_value=2)
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # print(uploaded_image)
        image = Image.open(uploaded_image)
        image_cv2 = np.array(image.convert('RGB'))
        st.image(image_cv2)
        
        output_image = processing(image_cv2, threshold, batch_size)
        # blue = np.zeros(image_cv2.shape)
        # for i in range(0, len(output_image)):
        #     for j in range(0, len(output_image[0])):
        #         if output_image[i][j] != 0:
        #             image_cv2 = cv2.circle(image_cv2, (i, j), 5, (0,255,0), 1)
        image_cv2[output_image > 0, 1] = 255
        
        st.image(output_image)
        st.image(image_cv2)
        

if __name__ == "__main__":
    main()
