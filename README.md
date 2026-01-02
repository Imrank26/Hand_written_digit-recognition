

# 🔢 AI Handwritten Digit Recognizer

A robust, end-to-end Deep Learning project that recognizes handwritten digits from real-world photographs. This project uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset and is deployed as a web application using **Streamlit**.

## 🌟 Key Features

* **High Accuracy CNN:** Built with TensorFlow/Keras using Batch Normalization and Dropout for better generalization.
* **Real-World Robustness:** Uses **Otsu’s Binarization** and **Bounding Box Cropping** to handle different lighting, pen colors, and image sizes.
* **Data Augmentation:** The model is trained on rotated and zoomed images to recognize varied handwriting styles.
* **Interactive UI:** A clean Streamlit interface for image uploads and real-time predictions.
* **Cloud Ready:** Configured to run on Google Colab with a public URL via `pyngrok`.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (for advanced image preprocessing)
* **Web Framework:** Streamlit
* **Deployment/Tunneling:** pyngrok

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mnist-digit-recognizer.git
cd mnist-digit-recognizer

```

### 2. Install Dependencies

```bash
pip install tensorflow streamlit opencv-python-headless pillow numpy pyngrok

```

### 3. Run the App

```bash
streamlit run app.py

```

---

## ☁️ How to Run on Google Colab (with Ngrok)

If you are using the provided `.ipynb` notebook:

1. **Upload Files:** Upload `app.py` and `mnist_model.h5` to your Colab session.
2. **Get Ngrok Token:** Sign up at [ngrok.com](https://ngrok.com/) and copy your Authtoken.
3. **Run the Tunneling Cell:**

```python
from pyngrok import ngrok
import os

# Set your token
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")

# Run Streamlit in background
os.system("nohup streamlit run app.py --server.port 8501 &")

# Open Tunnel
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url.public_url}")

```

---

## 🖼️ Project Output & Logic

The application follows a 3-step pipeline to ensure high-confidence predictions:

1. **Image Inversion:** Converts real-world "dark-on-light" photos to MNIST-style "white-on-black".
2. **Otsu Binarization:** Automatically removes background noise (like paper lines) by finding the optimal contrast threshold.
3. **Centering:** Crops the digit to its bounding box and adds padding to match the  training format.

### **Sample Result:**

* **Input:** A photo of a blue ink '6' on lined paper.
* **AI View:** A clean, centered white '6' on a black background.
* **Prediction:** Digit 6 (Confidence: 99.8%).

---

## 👤 Author

**Mohammed Imran Khattal**

* Fresher Python Developer


**Would you like me to generate a `requirements.txt` file for you so that users can install all libraries with a single command?**
