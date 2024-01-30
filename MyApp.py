import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Set page title and favicon
st.set_page_config(
    page_title="Animal Image Detection",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add some space at the top
st.title("Animal Image Detection 🦄")

# Display the sidebar with a light gray background
st.sidebar.title("About this App")
st.sidebar.write("This app uses a pre-trained deep learning model to detect different species of animals in images.")
st.sidebar.write("It supports both image uploads and live camera input.")

# Add a separator line
st.sidebar.markdown("---")

st.sidebar.title("Instructions")
st.sidebar.write(
    """
    1. Choose an input method: either upload an image or use the camera.
    2. If uploading, select an image file with an animal.
    3. If using the camera, make sure an animal is in the frame.
    4. Click the 'Capture' button to take a photo.
    5. The app will display the image and provide the prediction result.
    """
)

# Add a separator line
st.sidebar.markdown("---")

st.sidebar.title("Author")
st.sidebar.write("Created by Quinten, Wout, Ybe")

# Choose between file upload or camera input with a subtle background color
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"], key="input_method", help="Select the input method")

# Initialize image variable
image = None

# Function to capture image using button
def capture_image():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Function for batch processing
def process_images(images):
    predictions = []
    for img in images:
        img_array = np.asarray(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert PIL Image to BGR format
        img_array = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_img_array
        prediction = model.predict(data)
        predictions.append(prediction)
    return predictions

# Error handling for file upload and camera capture
try:
    if option == "Upload Image":
        # Upload image through Streamlit
        uploaded_files = st.file_uploader("Choose images of animals...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            # Process images in batch
            images = [Image.open(file).convert("RGB") for file in uploaded_files]
            predictions = process_images(images)

            # Display results for each image
            for i, (img, pred) in enumerate(zip(images, predictions)):
                st.image(img, caption=f"Uploaded Image {i+1}", width=300)
                index = np.argmax(pred)
                confidence_score = pred[0][index]

                # Display predicted class or "Unknown" if confidence is under 80%
                class_name = class_names[index] if confidence_score >= 0.8 else "..Unknown"

                st.subheader(f"Prediction Result for Image {i+1}:")
                st.write(f"Predicted Class: {class_name[2:]}", unsafe_allow_html=True, key=f'predicted_class_{i}')
                st.write(f"Confidence Score: {confidence_score:.2%}")

                # Display a message if confidence is under 80%
                if confidence_score < 0.8:
                    st.warning("The picture cannot be defined accurately or it is not an animal.")

                # Plot the prediction probabilities
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(class_names, pred[0], color='blue', width=0.5)
                ax.set_ylabel('Probability')
                ax.set_title(f'Prediction Probabilities for Image {i+1}')

                # Rotate x-axis labels to prevent overlap and increase spacing
                plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize='small')

                # Adjust layout to prevent shifting
                st.pyplot(fig, clear_figure=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Capture image using the camera button
if option == "Use Camera":
    if st.button("Capture"):
        try:
            image = capture_image()
            st.image(image, caption="Captured Image", width=300)
        except Exception as e:
            st.error(f"Error capturing image: {str(e)}")

# Check if image is defined before processing
if image is not None:
    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    # Display prediction and confidence score in Streamlit
    st.subheader("Prediction Result:")
    # Display predicted class or "Unknown" if confidence is under 80%
    class_name = class_names[index] if confidence_score >= 0.8 else "..Unknown"
    st.write(f"Predicted Class: {class_name[2:]}", unsafe_allow_html=True, key='predicted_class')
    st.write(f"Confidence Score: {confidence_score:.2%}")

    # Check if the detected class is "person"
    if "person" in class_name.lower():
        st.warning("A person is in the frame!")

    # Plot the prediction probabilities with adjusted label rotation and spacing
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size as needed
    ax.bar(class_names, prediction[0], color='blue', width=0.5)  # Adjust the width as needed
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')

    # Rotate x-axis labels to prevent overlap and increase spacing
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize='small')

    # Adjust layout to prevent shifting
    st.pyplot(fig, clear_figure=True)  # Use clear_figure to avoid duplication
