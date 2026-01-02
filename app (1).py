import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 # Import OpenCV

# --- Page Config ---
st.set_page_config(page_title="MNIST Real-Photo Fix", layout="centered")

# --- Load Model & Cache It ---
@st.cache_resource
def load_model():
    # Ensure mnist_model.h5 is uploaded to Colab files
    return tf.keras.models.load_model('mnist_model.h5')

try:
    model = load_model()
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}. Please make sure 'mnist_model.h5' is uploaded.")
    st.stop()

# --- Helper Function: Preprocess Image ---
def process_image(uploaded_file):
    # 1. Convert uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Read immediately as Grayscale
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # 2. THE CRITICAL STEP: Invert & ThresholdColors
    # cv2.THRESH_BINARY_INV turns dark pixels white, and light pixels black.
    # The '100' is the threshold value. Pixels darker than gray level 100 become white.
    # This helps remove the light blue paper lines.
    ret, img_inverted = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 3. Resize to 28x28
    img_resized = cv2.resize(img_inverted, (28, 28))
    
    # 4. Normalize (0-1 range) and Reshape for the model
    img_final = img_resized / 255.0
    img_final = img_final.reshape(1, 28, 28, 1)
    
    # Return both final data for prediction AND the visual image for display
    return img_final, img_resized

# --- Main UI ---
st.title("📸 Real-Photo Digit Recognizer")
st.write("This version inverts colors so real photos look like MNIST data.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Create two columns to show Before & After
    col1, col2 = st.columns(2)
    
    # Show original
    original_image = Image.open(uploaded_file)
    with col1:
        st.image(original_image, caption="Original Photo", use_container_width=True)
        
    # Process the image
    # Important: seek(0) resets file pointer so we can read it again in the function
    uploaded_file.seek(0) 
    img_for_model, img_for_display = process_image(uploaded_file)
    
    # Show processed image (What the AI sees)
    with col2:
        st.image(img_for_display, caption="AI View (Inverted)", use_container_width=True, clamp=True)

    st.markdown("---")

    # Predict Button
    if st.button("Predict Digit", type="primary"):
        with st.spinner("Analyzing..."):
            prediction = model.predict(img_for_model, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

        st.header(f"Prediction: {predicted_digit}")
        st.metric("Confidence", f"{confidence:.2f}%")
        
        if confidence < 60:
            st.warning("Low confidence. Try taking a photo with better lighting or thicker pen strokes.")