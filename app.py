# Install necessary packages
!apt-get install -y tesseract-ocr  # Keep Tesseract for comparison, if needed
!pip install streamlit opencv-python-headless pillow pyngrok ultralytics torch matplotlib easyocr

# Create and write the app.py file
code = '''
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
import easyocr
import matplotlib.pyplot as plt
import tempfile

# Streamlit app
def main():
    st.set_page_config(page_title="YOLO Object & Text Detection", layout="wide")
    st.title("🚀 YOLO Object Detection & Text Recognition Simulation")

    # Set device for inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"**Device Used**: {device}", unsafe_allow_html=True)

    # Load custom YOLO models
    model_product = YOLO('best.pt').to(device)  # Product detection
    model_fruit = YOLO('Quality.pt').to(device)  # Fruit quality detection

    # Classes for products
    target_classes = ['Beverages', 'Chocolates', 'Dairy-Product', 'Detergent', 'Grains', 'Skincare Product', 'Snacks']

    # Functions to adjust image quality
    def adjust_image_properties(image, brightness=0, contrast=30, saturation=1.0, sharpness=1.0):
        img = np.int16(image)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        return np.uint8(img)

    # Object detection function
    def detect_objects(image, model):
        resized_image = cv2.resize(image, (640, 640))  # Resize for accuracy/speed balance
        results = model(resized_image)

        labels = results[0].boxes.cls.cpu().numpy()  # Class labels
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

        detected_objects = []
        for box, label, confidence in zip(boxes, labels, confidences):
            if confidence > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = box
                class_name = model.names[int(label)]
                detected_objects.append((x1, y1, x2, y2, class_name, confidence))

        return resized_image, detected_objects

    # Text detection function using EasyOCR
    def detect_text(image):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image, detail=1)
        return result  # List of tuples (bounding box, text, confidence)

    # Section for product image input (both capture and upload)
    st.header("📸 Capture or Upload Product Image")

    # Camera input
    camera_product = st.camera_input("Take a picture of the product")
    image_product = None

    if camera_product:
        image_product = Image.open(camera_product)
        st.image(image_product, caption='Captured Product Image', use_column_width=True)

    # File uploader for product image
    uploaded_product_file = st.file_uploader("Or upload an image of the product", type=["jpg", "jpeg", "png"])

    if uploaded_product_file:
        image_product = Image.open(uploaded_product_file)
        st.image(image_product, caption='Uploaded Product Image', use_column_width=True)

    if image_product:
        # Convert to NumPy array for processing
        image_product = np.array(image_product)

        # Sidebar options for image adjustment
        st.sidebar.header("🎨 Adjust Product Image Settings")
        brightness_product = st.sidebar.slider('Brightness', -100, 100, 0)
        contrast_product = st.sidebar.slider('Contrast', -100, 100, 30)
        saturation_product = st.sidebar.slider('Saturation', 0.5, 2.0, 1.0)
        sharpness_product = st.sidebar.slider('Sharpness', 0.5, 2.0, 1.0)
        preprocessed_image_product = adjust_image_properties(image_product, brightness_product, contrast_product, saturation_product, sharpness_product)

        # Detect objects in product image
        st.write("🔍 **Detecting Products...**")
        detected_image_product, detected_objects_product = detect_objects(preprocessed_image_product, model_product)

        # Display processed image with bounding boxes
        st.image(detected_image_product, caption='Processed Image with Detections', use_column_width=True)

        # Show detected classes and confidence scores
        st.subheader("Detected Products")
        product_counts = {class_name: 0 for class_name in target_classes}
        for box in detected_objects_product:
            x1, y1, x2, y2, class_name, confidence = box
            confidence_percentage = confidence * 100
            st.markdown(f"**Class:** {class_name}, **Confidence:** {confidence_percentage:.2f}%")
            product_counts[class_name] += 1

        # Text detection
        st.write("🔍 **Detecting Text...**")
        detected_text = detect_text(preprocessed_image_product)
        if detected_text:
            st.subheader("Detected Text")
            for bbox, text, confidence in detected_text:
                st.markdown(f"**Text:** {text}, **Confidence:** {confidence:.2f}%")

        # Visualize product counts
        st.subheader("📊 Product Category Distribution")
        fig, ax = plt.subplots()
        ax.bar(product_counts.keys(), product_counts.values(), color='orange')
        ax.set_title('Product Counts')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # File uploads for fruit image
    uploaded_fruit_file = st.file_uploader("Upload Fruit Image", type=["jpg", "jpeg", "png"], key='fruit')

    if uploaded_fruit_file:
        # Display fruit image
        image_fruit = Image.open(uploaded_fruit_file)
        image_fruit = np.array(image_fruit)
        st.image(image_fruit, caption='Original Fruit Image', use_column_width=True)

        # Sidebar options for fruit image adjustment
        st.sidebar.header("🍎 Adjust Fruit Image Settings")
        brightness_fruit = st.sidebar.slider('Brightness', -100, 100, 0, key='brightness_fruit')
        contrast_fruit = st.sidebar.slider('Contrast', -100, 100, 30, key='contrast_fruit')
        preprocessed_image_fruit = adjust_image_properties(image_fruit, brightness_fruit, contrast_fruit)

        # Detect fruit quality
        st.write("🔍 **Detecting Fruit Quality...**")
        detected_image_fruit, detected_objects_fruit = detect_objects(preprocessed_image_fruit, model_fruit)

        # Display processed fruit image with detections
        st.image(detected_image_fruit, caption='Processed Fruit Image with Detections', use_column_width=True)

        # Show detected classes and confidence levels
        st.subheader("Fruit Quality Detection")
        fruit_classes = ['bad apple', 'bad banana', 'bad orange', 'good apple', 'good banana', 'good orange']
        for box in detected_objects_fruit:
            x1, y1, x2, y2, class_name, confidence = box
            if class_name in fruit_classes:
                confidence_percentage = confidence * 100
                st.markdown(f"**Class:** {class_name}, **Confidence:** {confidence_percentage:.2f}%")

if __name__ == "__main__":
    main()
'''

# Save code to app.py
with open("app.py", "w") as file:
    file.write(code)

# Function to start ngrok and run the Streamlit app
from pyngrok import ngrok
import time

def start_ngrok(port):
    ngrok.set_auth_token("2nZQ7ORczmcsrqAhPJVowkPuDfZ_3TN4T91btcThPtiTu1bgt")  # Replace with your actual ngrok auth token
    public_url = ngrok.connect(port)
    return public_url

# Run the Streamlit app
get_ipython().system_raw('streamlit run app.py --server.port 8501 &')

# Allow time for the app to start before checking ngrok
time.sleep(5)  # Wait for 5 seconds for Streamlit to initialize

# Start ngrok tunnel and get public URL
public_url = start_ngrok(8501)  # Connect ngrok to the correct port
print(f"Streamlit app is available at: {public_url}")
