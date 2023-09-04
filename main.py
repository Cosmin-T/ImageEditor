# Import Requried Libraries
import streamlit as st
from frontend.editor import *
from frontend.file_uploader import *
import io
from PIL import Image
import numpy as np
import base64

# Configure Streamlit page settings
st.set_page_config(page_title="Image Editor", layout="wide")

# Apply custom styling using HTML/CSS
st.markdown("""
<style>
    body {
        background-color: #FFFFFF;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #999;
        font-weight: 500;
        transition: color 0.3s;
    }
    .stApp {
        background: linear-gradient(270deg, #0e1017 40%, #434456 170%);
        border-radius: 10px;
        box-shadow: 3px 3px 20px rgba(0, 0, 0, 0.3);
        padding: 20px;
    }
    .stButton>button {
        background-color: #8B0000;
        color: #FFF;
        border: none;
        border-radius: 12px;
    }
    .stButton>button:hover {
        background-color: #5E0000;
    }
    .stSlider>div>div>div {
        background-color: #8B0000 !important;
    }
    .shining-title {
        font-size: 3em;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.8), transparent);
        background-size: 250% 100%;
        background-repeat: no-repeat;
        color: #000;
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        animation: shineTitle 8s infinite;
        transition: font-size 0.3s; /* Added transition */
    }
    .shining-title:hover {
        font-size: 3.5em; /* Adjusted font size on hover */
    }

    @keyframes shineTitle {
        0% { background-position: 100% 0; }
        80% { background-position: 50% 0; }
        100% { background-position: 100% 0; }
    }

        /* Zoom in/out effect */
    .zoom:hover {
        transform: scale(2);
        transition: transform .25s;
    }
</style>
""", unsafe_allow_html=True)


# Create a list to store image states for UNDO functionality
image_states = []

# Add a shining, fading title using HTML/CSS
st.markdown("<div class='shining-title'>Editor</div>", unsafe_allow_html=True)
st.sidebar.markdown('---')

# Sidebar content: Upload and Edit sections
st.sidebar.markdown("## Upload and Edit")
st.sidebar.markdown('---')

# Load the uploaded image using a custom function from the frontend module
uploaded_image = load_image()
if uploaded_image:
    # If 'image_states' is not present in session state, initialize it as an empty list
    if 'image_states' not in st.session_state:
        st.session_state.image_states = []

    # Store the current image state in the session state list for UNDO functionality
    st.session_state.image_states.append(uploaded_image.copy())
    st.sidebar.markdown('---')

    # Perform image editing using a custom function from the frontend module
    uploaded_image = edit_image(uploaded_image)

    # Convert the edited image to a PIL Image if it's not already in numpy array format
    if not isinstance(uploaded_image, np.ndarray):
        uploaded_image = np.array(uploaded_image)
    uploaded_image = Image.fromarray(uploaded_image)

    # Convert the PIL Image to TIFF format and store in a buffer
    buffer = io.BytesIO()
    uploaded_image.save(buffer, format='TIFF')
    image_data = buffer.getvalue()

    # Convert the TIFF image data to base64 encoding for displaying in the web app
    img_base64 = base64.b64encode(image_data).decode()

    # Display the edited image with a zoom effect using Markdown and HTML
    st.sidebar.markdown('---')
    st.markdown(f'<div class="zoom"><img src="data:image/png;base64,{img_base64}" alt="Edited Image" style="width:100%"/></div>', unsafe_allow_html=True)

    # Add a download button to download the edited image
    st.sidebar.download_button(
        label='Download',
        data=buffer.getvalue(),
        file_name='edited_image.TIFF',
        mime='image/png'
    )