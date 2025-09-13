import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# -------------------------------
# Load your trained DenseNet121 model
# -------------------------------
model = load_model("/home/mohamed-elgohary/MyWorkSpace/Teeth_diseases_Classification/Model_from_Scratch/Model/model_Teeth.keras")

# Define image size (adjust to your training input size)
IMG_SIZE = (224, 224)

# Define class names
class_names = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Dental AI - Teeth Classification", layout="centered")

st.title("🦷 Dental AI: Teeth Classification")
st.write(
    """
    Upload a dental image and our AI model will classify it into one of the **7 categories**:  
    - CaS  
    - CoS  
    - Gum  
    - MC  
    - OC  
    - OLP  
    - OT  
    """
)

# Upload image
uploaded_file = st.file_uploader("📤 Upload a dental image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))

    # Display results
    st.subheader("✅ Prediction Results")
    st.write(f"**Predicted Class:** {class_names[predicted_index]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Optional: Show confidence for all classes
    st.bar_chart(dict(zip(class_names, prediction[0])))
