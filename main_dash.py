import numpy as np
import cv2
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import base64
from io import BytesIO
import os

# Initialize the Dash app
app = dash.Dash(__name__)

# Model paths
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Define the layout of the Dash interface
app.layout = html.Div([
    html.H1("Image Colorization"),
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload Image'),
        multiple=False
    ),
    html.Div(id='output-image-upload'),
])

# Function to process the uploaded image
def process_image(image_content):
    # Decode the uploaded image
    decoded_image = base64.b64decode(image_content.split(',')[1])
    np_img = np.asarray(bytearray(decoded_image), dtype=np.uint8)
    bw_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

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

    # Convert images to base64
    _, bw_encoded = cv2.imencode('.jpeg', bw_image)
    bw_base64 = base64.b64encode(bw_encoded).decode('utf-8')

    _, colorized_encoded = cv2.imencode('.jpeg', colorized)
    colorized_base64 = base64.b64encode(colorized_encoded).decode('utf-8')

    return bw_base64, colorized_base64

# Callback to update the image display after upload
@app.callback(
    Output('output-image-upload', 'children'),
    [Input('upload-image', 'contents')]
)
def update_output(image_content):
    if image_content is not None:
        bw_base64, colorized_base64 = process_image(image_content)

        return html.Div([
            html.Div([
                html.H3("Black and White Image"),
                html.Img(src=f"data:image/jpeg;base64,{bw_base64}", style={"width": "100%", "height": "auto"})
            ]),
            html.Div([
                html.H3("Colorized Image"),
                html.Img(src=f"data:image/jpeg;base64,{colorized_base64}", style={"width": "100%", "height": "auto"})
            ])
        ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
