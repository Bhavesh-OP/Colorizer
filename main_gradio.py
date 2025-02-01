import os
import requests
import numpy as np
import cv2
import gradio as gr

# Define the URL and the destination path
url = "https://uca88b8a467313dbd32503f32cd9.dl.dropboxusercontent.com/cd/0/get/CjN6RWs9aVScWRqfEDUgCKmow0rDtc9hFcQk70avprr05CtMpGYq7KgktcJBEIFDxuAAtLApl-DSVhKyzgDtUF0geVuMTB0aentyjOVKX8WLqqRrrNY72AEcV4mixOLDgI5xZE9fBWwrRKQfy6nVOGKv/file?dl=1"
destination_folder = "models"
destination_file = os.path.join(destination_folder, "colorization_release_v2.caffemodel")

# Create the directory if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

try:
    # Download the file
    print("Downloading model...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for failed downloads

    # Save the file
    with open(destination_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Download completed: {destination_file}")

except requests.exceptions.RequestException as e:
    print(f"Download failed: {e}")

def colorize_image(image):
    prototxt_path = 'models/colorization_deploy_v2.prototxt'
    model_path = 'models/colorization_release_v2.caffemodel'
    kernel_path = 'models/pts_in_hull.npy'
    
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)

    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    bw_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1,2,0))

    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:,:, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")
    
    return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

iface = gr.Interface(
    fn=colorize_image,
    inputs="image",
    outputs="image",
    title="Black & White Image Colorization",
    description="Upload a black and white image, and this AI will colorize it automatically.",
    theme="default"
)
iface.launch()
