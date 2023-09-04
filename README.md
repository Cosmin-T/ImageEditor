# Image Editor App using Streamlit


This repository contains a Python application built using the Streamlit library that empowers users to upload images and apply a diverse range of image editing functionalities. The app offers an intuitive and user-friendly interface for editing images, incorporating features such as resizing, rotation, filters, exposure adjustments, color enhancements, cropping, and more.

## Features

- **Image Uploading**: Users can effortlessly upload images in various formats (jpg, png, gif, heif, jpeg, tiff) using the file uploader widget available in the sidebar.
- **Image Editing**: The app offers an extensive array of editing operations that users can apply to the uploaded image, including:
	- Resizing the image to specific dimensions.
	- Rotating the image by a specified degree.
	- Applying diverse image filters such as blur, contour, sharpen, and more.
	- Adjusting exposure, brilliance, highlights, shadows, contrast, brightness, black point, saturation, vibrancy, warmth, tint, sharpness, and noise reduction.
	- Applying simple super-resolution techniques to enhance image quality.
	- Adding automatic enhancements for color, contrast, and sharpness.
	- Cropping the image to a specific resolution.
- **Undo Functionality**: The app incorporates an "UNDO" button, allowing users to effortlessly revert to previous image states during the editing process.
- **Image Display**: The edited image is prominently displayed with a zoom effect in the main app interface, and users have the option to download the edited image in TIFF format.

## Files

- `editor.py`: This file contains functions for various image editing operations utilizing the PIL (Pillow), OpenCV (cv2), and NumPy libraries. It also includes functions for implementing auto-enhancements, super-resolution, sharpening, and cropping.
- `file_uploader.py`: This file provides functions for uploading images and displaying relevant image information in the sidebar.
- `main.py`: The main app script imports the necessary libraries, configures the Streamlit page, and defines the layout of the Streamlit app. It seamlessly integrates image uploading, editing, and display functionalities from the other files.

## Usage

1. Install the required libraries using the command: `pip install -r requirements.txt`.
2. Run the app by executing: `streamlit run main.py`.
3. Upload an image using the file uploader widget in the sidebar.
4. Utilize the various editing functionalities in the sidebar to apply desired image adjustments.
5. Optionally, use the "UNDO" button to revert to previous image states.
6. Download the edited image using the download button provided in the sidebar.

## Customization

- You can easily customize the app by modifying functions within the `editor.py` file to add or alter image editing operations.
- Adjust the layout and styling of the app by modifying the HTML/CSS code in the `main.py` file.

## Notes

- This app has been developed using Streamlit along with Python's imaging libraries (PIL, OpenCV, NumPy).
- The provided code is intended for educational and illustrative purposes and can be extended to suit specific requirements.
- Feel free to explore, customize, and use the app as a foundation for creating more advanced image editing applications.
