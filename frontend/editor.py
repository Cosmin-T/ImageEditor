# Import required libraries
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from skimage import exposure
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

# Function to convert PIL Image to numpy array
def img_to_np(img):
    return np.array(img)

# Function to convert numpy array to PIL Image
def np_to_img(arr):
    return Image.fromarray(arr.astype('uint8'))

# Function to resize the image
def resize(img):
    # Use the current image dimensions for the default values of the number inputs.
    width = st.sidebar.number_input("Width", value=img.width)
    height = st.sidebar.number_input("Height", value=img.height)
    return img.resize((width, height))

# Function to rotate te image
def rotation(img):
    degree = st.sidebar.slider("Rotation Degree", 0, 360, 0, 1)
    return img.rotate(degree)

# Function to apply various cv2 filters to the image
def filters(img):
    filter_type = st.sidebar.selectbox("Filters", options=("None", "Blur", "Contour", "Detail", "Edge Enhance", "Edge Enhance +", "Find Edges", "Sharpen", "Emboss", "Smooth", "Smooth +"))
    if filter_type == "Blur":
        return img.filter(ImageFilter.BLUR)
    elif filter_type == "Contour":
        return img.filter(ImageFilter.CONTOUR)
    elif filter_type == "Detail":
        return img.filter(ImageFilter.DETAIL)
    elif filter_type == "Edge Enhance":
        return img.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_type == "Edge Enhance +":
        return img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif filter_type == "Find Edges":
        return img.filter(ImageFilter.FIND_EDGES)
    elif filter_type == "Sharpen":
        return img.filter(ImageFilter.SHARPEN)
    elif filter_type == "Emboss":
        return img.filter(ImageFilter.EMBOSS)
    elif filter_type == "Smooth":
        return img.filter(ImageFilter.SMOOTH)
    elif filter_type == "Smooth +":
        return img.filter(ImageFilter.SMOOTH_MORE)
    else:
        return img

# Function to adjust exposure of the image
def exposure(img):
    level = st.sidebar.slider("Exposure", 0, 200, 100)
    if level == 100:
        return img
    img_array = np.array(img)
    exposure_adjusted = cv2.convertScaleAbs(img_array, alpha=level / 100, beta=0)
    return Image.fromarray(exposure_adjusted)

# Function to adjust brilliance of the image
def brilliance(img):
    level = st.sidebar.slider("Brilliance", -100, 100, 0)
    if level == 0:
        return img
    img_array = np.array(img)
    factor = (259 * (level + 255)) / (255 * (259 - level))
    brilliance_adjusted = cv2.convertScaleAbs(img_array, alpha=factor, beta=0)
    return Image.fromarray(brilliance_adjusted)

# Function to adjust highlights of the image
def highlights(img):
    level = st.sidebar.slider("Highlights", 0, 100, 50)
    if level == 50:
        return img
    img_array = np.array(img)
    highlights_adjusted = cv2.convertScaleAbs(img_array, alpha=1, beta=(level - 50) / 50)
    return Image.fromarray(highlights_adjusted)

# Function to adjust shadows of the image
def shadows(img):
    level = st.sidebar.slider("Shadows", 0, 100, 50)
    if level == 50:
        return img
    img_array = np.array(img)
    shadows_adjusted = cv2.convertScaleAbs(img_array, alpha=1, beta=(100 + level) / 50)
    return Image.fromarray(shadows_adjusted)

# Function to adjust contrast of the image
def contrast(img):
    level = st.sidebar.slider("Contrast", 1, 100, 50)
    if level == 50:
        return img
    img_array = np.array(img)
    contrast_adjusted = cv2.convertScaleAbs(img_array, alpha=(level - 50) / 50 + 1, beta=0)
    return Image.fromarray(contrast_adjusted)

# Function to adjust brightness of the image
def brightness(img):
    level = st.sidebar.slider("Brightness", 1, 100, 50)
    if level == 50:
        return img
    img_array = np.array(img)
    brightness_adjusted = cv2.convertScaleAbs(img_array, alpha=1, beta=level - 50)
    return Image.fromarray(brightness_adjusted)

# Function to adjust black point of the image
def black_point(img):
    level = st.sidebar.slider("Black Point", 0, 100, 50)
    if level == 50:
        return img
    img_array = np.array(img).astype(np.uint8)
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    scale_factor = 255 / (max_val - min_val + level)
    black_point_adjusted = np.clip((img_array - min_val) * scale_factor, 0, 255).astype(np.uint8)
    return Image.fromarray(black_point_adjusted)

# Function to adjust saturation of the image
def saturation(img):
    level = st.sidebar.slider("Saturation", 0, 200, 100)
    if level == 100:
        return img
    enhancer = ImageEnhance.Color(img)
    saturation_adjusted = enhancer.enhance(level / 100)
    return saturation_adjusted

# Function to adjust vibrancy of the image
def vibrancy(img):
    level = st.sidebar.slider("Vibrancy", 0, 200, 100)
    if level == 100:
        return img
    enhancer = ImageEnhance.Color(img)
    vibrancy_adjusted = enhancer.enhance(level / 100)
    return vibrancy_adjusted

# Function to adjust warmth of the image
def warmth(img):
    level = st.sidebar.slider("Warmth", -100, 100, 0)
    if level == 0:
        return img
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    warming_filter = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)
    warmth_adjusted = cv2.addWeighted(img_array, 1, warming_filter, level/100, 0)
    return Image.fromarray(cv2.cvtColor(warmth_adjusted, cv2.COLOR_BGR2RGB))

# Function to adjust tint of the image
def tint(img):
    level = st.sidebar.slider("Tint", -100, 100, 0)
    if level == 0:
        return img
    img_array = np.array(img)
    tint_adjusted = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    tint_adjusted[:, :, 0] = np.clip(tint_adjusted[:, :, 0] + level, 0, 179)
    tint_adjusted = cv2.cvtColor(tint_adjusted, cv2.COLOR_HSV2RGB)
    return Image.fromarray(tint_adjusted)

# Function to adjust sharpness of the image
def sharpness(img):
    level = st.sidebar.slider("Sharpness", 0, 100, 50)
    if level == 50:
        return img
    enhancer = ImageEnhance.Sharpness(img)
    sharpness_adjusted = enhancer.enhance(level / 50)
    return sharpness_adjusted

# Function to adjust noise reduction of the image
def noise_reduction(img):
    level = st.sidebar.slider("Noise Reduction", 0, 100, 0)
    if level == 0:
        return img
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    noise_reduction_adjusted = cv2.GaussianBlur(img_array, (21, 21), level)
    return Image.fromarray(cv2.cvtColor(noise_reduction_adjusted, cv2.COLOR_BGR2RGB))

# Function to adjust simpple super resolution of the image
def simple_super_resolution(img):
    # Check if the "Simple Super-Resolution" checkbox is checked in the sidebar
    apply_super_res = st.sidebar.checkbox("Simple Super-Resolution", False)

    # If the checkbox is not checked, return the original image
    if not apply_super_res:
        return img

    # Get the super-resolution factor from the slider in the sidebar
    factor = st.sidebar.slider("Super-Resolution Factor", 1.0, 4.0, 1.0)

    # Convert the PIL Image to a numpy array
    img_array = np.array(img)

    # Check if the image is a color image (3 or 4 channels)
    if img_array.ndim == 3 and img_array.shape[-1] in [3, 4]:
        # Create an empty list to store rescaled channels
        channel_list = []

        # Iterate through each channel and apply rescaling
        for i in range(img_array.shape[2]):
            # Rescale the current channel using the specified factor
            scaled_channel = rescale(img_array[:,:,i], factor, anti_aliasing=False, mode='reflect')

            # Scale the pixel values to the range [0, 255] and convert to uint8
            channel_list.append((scaled_channel * 255).astype(np.uint8))

        # Stack the rescaled channels along the last dimension to create a color image
        img_rescaled = np.stack(channel_list, axis=2)
    else:
        # Rescale the image using the specified factor
        img_rescaled = (rescale(img_array, factor, anti_aliasing=False, mode='reflect') * 255).astype(np.uint8)

    # Convert the numpy array back to a PIL Image and return the rescaled image
    return Image.fromarray(img_rescaled)

# Function to add sharpening to the image
def sharpen_image(image, sharpen_factor=1.0, blur_kernel_size=(3, 3)):
    blurred = cv2.GaussianBlur(image, blur_kernel_size, 0)
    sharpened = cv2.addWeighted(image, 1 + sharpen_factor, blurred, -sharpen_factor, 0)
    return sharpened

# Function to add auto-enhancment to the image using sharpen_image function
def red_enhance(img, contrast_factor=-0.07, brightness_factor=0.15, sharpen_factor=0.5):
    try:
        # Convert the input PIL Image to a numpy array
        img_array = np.array(img)

        # Print the shape of the image array for debugging
        print("Image array shape:", img_array.shape)
    except Exception as e:
        raise ValueError("Invalid image provided: {}".format(e))

    # Determine the number of color channels in the image
    num_channels = img_array.shape[-1]

    # Check if the image has the expected number of channels (3 for BGR, 4 for RGBA)
    if num_channels not in [3, 4]:
        raise ValueError("Input should be a 3-channel BGR or 4-channel RGBA image.")

    # Separate the image array into BGR channels (and alpha channel, if present)
    if num_channels == 4:
        img_bgr = img_array[:, :, :3]
        img_alpha = img_array[:, :, 3]
    else:
        img_bgr = img_array

    # Convert the BGR image to LAB color space for controlled adjustments
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Apply contrast and brightness adjustments to the L channel
    l = np.clip(l * contrast_factor + brightness_factor * 255, 0, 255).astype(np.uint8)

    # Merge the adjusted channels and convert back to BGR color space
    enhanced_lab = cv2.merge((l, a, b))
    img_bgr_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply sharpening to the enhanced BGR image
    img_bgr_sharpened = sharpen_image(img_bgr_enhanced, sharpen_factor)

    # Concatenate the sharpened BGR image with the alpha channel (if present)
    if num_channels == 4:
        img_array_sharpened = np.concatenate([img_bgr_sharpened, img_alpha[:, :, np.newaxis]], axis=2)
    else:
        img_array_sharpened = img_bgr_sharpened

    # Convert the numpy array back to a PIL Image and return the enhanced and sharpened image
    return Image.fromarray(img_array_sharpened.astype('uint8'))

# Function to add auto-enhancment to the image using sharpen_image function
def auto_enhance(img, contrast_factor=1.0, brightness_factor=-0.05, sharpen_factor=2.5):
    try:
        # Convert the input PIL Image to a numpy array
        img_array = np.array(img)

        # Print the shape of the image array for debugging
        print("Image array shape:", img_array.shape)
    except Exception as e:
        raise ValueError("Invalid image provided: {}".format(e))

    # Determine the number of color channels in the image
    num_channels = img_array.shape[-1]

    # Check if the image has the expected number of channels (3 for BGR, 4 for RGBA)
    if num_channels not in [3, 4]:
        raise ValueError("Input should be a 3-channel BGR or 4-channel RGBA image.")

    # Separate the image array into BGR channels (and alpha channel, if present)
    if num_channels == 4:
        img_bgr = img_array[:, :, :3]
        img_alpha = img_array[:, :, 3]
    else:
        img_bgr = img_array

    # Convert the BGR image to LAB color space for controlled adjustments
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Apply contrast and brightness adjustments to the L channel
    l = np.clip(l * contrast_factor + brightness_factor * 255, 0, 255).astype(np.uint8)

    # Merge the adjusted channels and convert back to BGR color space
    enhanced_lab = cv2.merge((l, a, b))
    img_bgr_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply sharpening to the enhanced BGR image
    img_bgr_sharpened = sharpen_image(img_bgr_enhanced, sharpen_factor)

    # Concatenate the sharpened BGR image with the alpha channel (if present)
    if num_channels == 4:
        img_array_sharpened = np.concatenate([img_bgr_sharpened, img_alpha[:, :, np.newaxis]], axis=2)
    else:
        img_array_sharpened = img_bgr_sharpened

    # Convert the numpy array back to a PIL Image and return the enhanced and sharpened image
    return Image.fromarray(img_array_sharpened.astype('uint8'))

# Function to crop the image to 1440x2560 pixels
def crop_to_1440x2560(img):
    # Define the target dimensions for cropping
    target_width = 1440
    target_height = 2560

    # Get the current width and height of the input image
    img_width, img_height = img.size

    # Check if the image dimensions are smaller than the target resolution
    if img_width < target_width or img_height < target_height:
        # Display a warning message if the image is too small for cropping
        st.warning("Image dimensions are smaller than the target resolution, please use Simple-Super Resolution")
        return img

    # Calculate the cropping coordinates to center the image
    left = (img_width - target_width) // 2
    top = (img_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    # Crop the image using the calculated coordinates
    cropped_img = img.crop((left, top, right, bottom))

    # Return the cropped image
    return cropped_img

# Function to apply all prior functions to the image
def edit_image(uploaded_image):
    # Add Auto Enhancements section in the sidebar
    st.sidebar.title("Auto Enhancements")

    # Create buttons for Auto-Enhance and Red-Enhance
    auto_enhance_button = st.sidebar.button("Auto-Enhance")
    red_enhance_button = st.sidebar.button("Red-Enhance")

    # Create a button to UNDO the last enhancement
    if st.sidebar.button("UNDO-Enhance"):
        # Check if there are previous image states in session state
        if len(st.session_state.image_states) > 1:
            # Remove the last image state and update uploaded_image
            st.session_state.image_states.pop()
            uploaded_image = st.session_state.image_states[-1]
        else:
            # Display a message if there's nothing to undo
            st.sidebar.markdown("Nothing to undo.")

    # Add a separator in the sidebar
    st.sidebar.markdown("---")

    # Create a button to crop the image to 1440x2560
    crop_button = st.sidebar.button("Crop to 1440x2560")

    # Add a separator in the sidebar
    st.sidebar.markdown("---")

    # Apply enhancements based on button clicks
    if red_enhance_button:
        uploaded_image = red_enhance(uploaded_image)
    elif auto_enhance_button:
        uploaded_image = auto_enhance(uploaded_image)

    # Apply super resolution to the image
    uploaded_image = simple_super_resolution(uploaded_image)

    # Check if crop is needed and apply it
    if crop_button:
        uploaded_image = crop_to_1440x2560(uploaded_image)

    # Apply resize to the image
    uploaded_image = resize(uploaded_image)

    # Apply various image adjustments
    uploaded_image = rotation(uploaded_image)
    uploaded_image = filters(uploaded_image)
    uploaded_image = exposure(uploaded_image)
    uploaded_image = brilliance(uploaded_image)
    uploaded_image = highlights(uploaded_image)
    uploaded_image = shadows(uploaded_image)
    uploaded_image = contrast(uploaded_image)
    uploaded_image = brightness(uploaded_image)
    uploaded_image = black_point(uploaded_image)
    uploaded_image = saturation(uploaded_image)
    uploaded_image = vibrancy(uploaded_image)
    uploaded_image = warmth(uploaded_image)
    uploaded_image = tint(uploaded_image)
    uploaded_image = sharpness(uploaded_image)
    uploaded_image = noise_reduction(uploaded_image)

    # Return the edited image
    return uploaded_image

