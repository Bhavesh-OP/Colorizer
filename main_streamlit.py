import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Load model and kernel points
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Streamlit UI for file upload
st.title('Image Colorization App')
st.write("Upload a black and white image to colorize it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image and convert to numpy array
    bw_image = np.array(Image.open(uploaded_file))

    # Check the dimensions and color channels of the input image
    if len(bw_image.shape) == 2:  # If the image is grayscale
        bw_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)

    # Normalize and process the image
    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Feed L channel into the model
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize ab channel to match original image size
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]

    # Combine L and ab to form the colorized image
    colorized_lab = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert the LAB image back to BGR
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)

    # Clip values to ensure they are within 0-255 range and convert to uint8
    colorized_bgr = np.clip(colorized_bgr * 255.0, 0, 255).astype("uint8")

    # Display the images
    st.image(bw_image, caption='Original Black and White Image', use_column_width=True)
    st.image(colorized_bgr, caption='Colorized Image', use_column_width=True)
