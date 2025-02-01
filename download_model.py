import os
import requests
import subprocess

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

    # Run the main_gradio.py script
    print("Running main_gradio.py...")
    subprocess.run(["python", "main_gradio.py"], check=True)

except requests.exceptions.RequestException as e:
    print(f"Download failed: {e}")

except subprocess.CalledProcessError as e:
    print(f"Error while running main_gradio.py: {e}")
