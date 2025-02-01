# Colorization Model Setup

This repository contains a deep learning-based colorization model. Follow the steps below to set up the required dependencies and run the application with different UI frameworks.

---

## **Installation Guide**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### **2. Install Dependencies**
Ensure you have Python installed (recommended version: 3.8+). Install the required dependencies:
```bash
pip install -r requirements.txt
```

### **3. Download the Model**
Before running any UI, first download the pre-trained model:
```bash
python download_model.py
```
This will download `colorization_release_v2.caffemodel` inside the `models/` folder.

---

## **Running the Application**
You can use different UIs to run the colorizer model. Choose one of the following options:

### **1. Normal Python UI**
Run the basic UI application:
```bash
python main_normal.py
```

### **2. Dash UI**
Run the Dash-based web application:
```bash
python main_dash.py
```

### **3. Streamlit UI**
Run the Streamlit-based web application:
```bash
streamlit run main_streamlit.py
```

### **4. Gradio UI**
Run the Gradio-based web interface:
```bash
python main_gradio.py
```

### **5. Flask Web UI**
If there is a Flask-based UI, run:
```bash
python main_flask.py
```

---

## **Additional Notes**
- Ensure you have a stable internet connection while downloading the model.
- If you encounter any issues with missing dependencies, manually install them using:
  ```bash
  pip install package-name
  ```
- If running on a GPU, ensure you have CUDA installed for better performance.

For any issues, feel free to raise an issue in the repository. 🚀

