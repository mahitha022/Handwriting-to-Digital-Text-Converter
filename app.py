import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Step 1: Load the Previously Saved Model
model = load_model("handwritten_text_recognition_model.keras")

# Step 2: Load and Preprocess a New Handwritten Image
# Example image path - Replace with your own image path
image_path = "image3.png"  # Make sure this is a 28x28 image or resize accordingly
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale

# Resize the image to 28x28 pixels if it is not already
if image.shape != (28, 28):
    image = cv2.resize(image, (28, 28))

# Normalize the image to range [0, 1]
image = image.astype("float32") / 255.0

# Reshape the image to match model input shape: (1, 28, 28, 1)
image = np.expand_dims(image, axis=-1)  # Add channel dimension
image = np.expand_dims(image, axis=0)   # Add batch dimension

# Step 3: Predict the Handwritten Digit/Character
predictions = model.predict(image)
predicted_class = np.argmax(predictions)

# Step 4: Display the Image and Prediction
print(f"Predicted Digit/Character: {predicted_class}")
