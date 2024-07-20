# Import Requried Libraries
import streamlit as st
from PIL import Image

# Function to create empty placeholders for image information in the sidebar
def empty_holder():
    # Create empty placeholders to display image information in the sidebar
    size = st.sidebar.empty()
    mode = st.sidebar.empty()
    format_ = st.sidebar.empty()
    return size, mode, format_

# Function to load and display an uploaded image
def load_image():
    # Display a file uploader widget for image uploads in the sidebar
    upload = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "gif", "heif", "jpeg", "tiff"])

    # Check if an image has been uploaded
    if upload is not None:
        # Open the uploaded image using PIL's Image class
        img = Image.open(upload)

        # Create empty placeholders for displaying image information
        size_holder, mode_holder, format_holder = empty_holder()

        # Display the size, mode, and format of the uploaded image
        size_holder.markdown(f"Size: {img.size}")
        mode_holder.markdown(f"Mode: {img.mode}")
        format_holder.markdown(f"Format: {img.format}")

        # Return the opened image
        return img
    else:
        # Return None if no image has been uploaded
        return None

