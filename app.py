import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
import easyocr
import matplotlib.pyplot as plt

# Streamlit app
def main():
    st.set_page_config(page_title="YOLO Object & Text Detection", layout="wide")
    st.title("🚀 YOLO Object Detection & Text Recognition Simulation")

    # Set device for inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"**Device Used**: {device}")

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
        preprocessed_image_product = adjust_image_properties(image_product, brightness_product, contrast_product)

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

if __name__ == "__main__":
    main()
