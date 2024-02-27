import streamlit as st
import tensorflow as tf
from PIL import Image
import os
from io import BytesIO
import base64
import zipfile
import cv2
import numpy as np
# Add the path to the 'src' directory to the system path
import sys
sys.path.append(os.path.abspath('src'))
import utils


# Constants and thresholds
threshold_lap_var = 100
threshold_entropy = 3.5
threshold_mean_gradient = 25
threshold_color_consistency = 50
threshold_light_misalignment = 30
dl_weight = 0.8
cv_weight = 0.2
threshold = 0.5

def main():
    st.set_page_config(layout="wide")
    
    # Add logo in the top left corner
    # Replace with the path to your logo image
    logo_image = Image.open(utils.logo_path)
    st.image(logo_image, width=150, use_column_width=False)

    # Add title in the middle
    st.markdown("<h1 style='text-align: center;font-size: 40px;'>ImageAuthenticator</h1>",
                unsafe_allow_html=True)
 
    css = """
    <style>
        /* Style for tabs */
         .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:20px;
        font-weight: bold;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size:20px;
            background-color: #54A0E8;  /* Set the default tab background color */
            padding: 8px;
            color: white;
            transition: background-color 0.3s ease;
        }

        .stTabs [data-baseweb="tab-list"] button:hover {
            background-color: #2A92D5;  /* Set the hover tab background color */
        }

        /* Style for container */
        .stTabs [data-baseweb="tab-list"]{
            background-color: #3182CE;
            padding: 10px;
            margin: 5px;
        }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

        # Create a container
    with st.container():
        # Create tabs inside the container
        tabs = st.tabs([" Upload and Predict", "ℹ️ Help"])
        # Content for Tab 1
        with tabs[0]:
            upload_and_predict()

        # Content for Tab 2
        with tabs[1]:
            show_help()


def upload_and_predict():
    # Upload and display multiple images
    uploaded_files = st.file_uploader('Choose images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files is not None:
        # Create empty lists to store the results
        images = []
        image_names = []
        captions = []
        prediction_messages = []

        for uploaded_file in uploaded_files:
            st.markdown("<p style='display: none;'>Filename: {}</p>".format(uploaded_file.name), unsafe_allow_html=True)
            st.markdown("<p style='display: none;'>Size: {} bytes</p>".format(uploaded_file.size), unsafe_allow_html=True)

            # Image preprocessing
            image_tensor = tf.io.decode_image(uploaded_file.read(), channels=3)
            image_tensor = tf.image.resize(image_tensor, (32, 32))
            image_tensor = tf.expand_dims(image_tensor, axis=0)

            # Load the trained model
            model = tf.keras.models.load_model(utils.vgg_final_model_path)

            # Make predictions
            prediction = model.predict(image_tensor)
            prediction_message = "It is a real image." if prediction[0][0] == 1 else "It is an AI-generated image."

            image = Image.open(uploaded_file)

            # Convert the image to OpenCV format
            image_array = np.array(image)
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            lap_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            entropy = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            entropy = entropy[entropy > 0]
            entropy = -np.sum((entropy / np.sum(entropy)) * np.log2(entropy / np.sum(entropy)))
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            mean_gradient = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            color_consistency = np.mean(np.std(image_array, axis=(0, 1)))
            light_misalignment = np.std(image_array, axis=(0, 1)).mean()

            # Define threshold values for each metric
            threshold_lap_var = 100
            threshold_entropy = 3.5
            threshold_mean_gradient = 25
            threshold_color_consistency = 50
            threshold_light_misalignment = 30

            real_reasons = []
            fake_reasons = []
            if lap_var < threshold_lap_var:
                fake_reasons.append("Low variance in Laplacian: The image has a low level of variation in the details and edges, suggesting it may be a generated image.")
            else:
                real_reasons.append("High variance in Laplacian: The image has a high level of variation in the details and edges, indicating it is likely a real image.")

            if entropy > threshold_entropy:
                fake_reasons.append("High entropy: The image has a high level of randomness or complexity, which is often associated with generated images.")
            else:
                real_reasons.append("Low entropy: The image has a low level of randomness or complexity, suggesting it is more likely a real image.")

            if mean_gradient < threshold_mean_gradient:
                fake_reasons.append("Low mean gradient: The image has a low average gradient, indicating a lack of pronounced changes in intensity and texture, which is often observed in generated images.")
            else:
                real_reasons.append("High mean gradient: The image has a high average gradient, suggesting pronounced changes in intensity and texture, characteristic of real images.")

            if color_consistency < threshold_color_consistency:
                fake_reasons.append("Low color consistency: The image exhibits low consistency in colors across different regions, which is commonly observed in generated images.")
            else:
                real_reasons.append("High color consistency: The image shows a high level of consistency in colors across different regions, indicating it is more likely a real image.")

            if light_misalignment > threshold_light_misalignment:
                fake_reasons.append("High light misalignment: The lighting in the image appears to be inconsistent or misaligned, which is often a characteristic of generated images.")
            else:
                real_reasons.append("Low light misalignment: The lighting in the image appears to be consistent and properly aligned, suggesting it is more likely a real image.")

            # Calculate the weighted average result
            dl_weight = 0.8  # Weight for the deep learning model
            cv_weight = 0.2  # Weight for the OpenCV analysis
            weighted_average = (dl_weight * prediction[0][0] + cv_weight * (1 if len(real_reasons) >= 3 else 0))

            threshold = 0.5  # Threshold value to determine if the image is fake or not

            if weighted_average >= threshold:
                classification_result = "Real"
            else:
                classification_result = "Fake"

            new_caption = f"{uploaded_file.name.split('.')[0]}_{classification_result}"

            images.append(image)
            image_names.append(uploaded_file.name)
            captions.append(new_caption)
            prediction_messages.append(prediction_message)

            history_item = {
                'image': image,
                'image_name': uploaded_file.name,
                'caption': new_caption,
                'prediction_message': prediction_message
            }
            
            update_history_log(history_item)
        # Initialize session variables
        position = st.session_state.get('position',0)
        # Display the results in two columns 
        num_images = len(images)
        if num_images > 0:
            col1, col2 = st.columns([1, 1])

            with col1:
                if(num_images == 1):
                    # image = images[0]
                    # image = image.resize((300,300))
                    st.image(images[0], use_column_width=True)
                    download_button_str = get_download_button_str(images[0], captions[0])
                    st.markdown(download_button_str, unsafe_allow_html=True)
                else:
                    prev, _ ,next = st.columns([2, 8.7, 1.3])
                    if next.button("Next"):
                        position +=1
                        if (position == num_images):
                           position = 0
                    if prev.button("Back"):
                        position -=1
                        if position == -1:
                            position = num_images - 1
                    st.session_state['position'] = position
                    st.image(images[position], use_column_width=True)
                    download_button_str = get_download_button_str(images[position], captions[position])
                    st.markdown(download_button_str, unsafe_allow_html=True)
               

            with col2:
                # Display the result and reasons
                st.markdown(f"<h3 style='text-align: center; font-weight: bold; font-size: 18px; padding-top: 50px;'>Result:</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 22px;'>It is a {classification_result} image.</p>", unsafe_allow_html=True)

                st.markdown("<h3 style='text-align: center; font-weight: bold; font-size: 18px; '>Reasons:</h3>", unsafe_allow_html=True)

                if classification_result == 'Real':
                    st.markdown("<ul style='text-align: justify;'>", unsafe_allow_html=True)
                    for reason in real_reasons:
                        st.markdown(f"<li style='font-size: 16px;'>{reason}</li>", unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)

                else:
                    st.markdown("<ul style='text-align: justify;'>", unsafe_allow_html=True)
                    for reason in fake_reasons:
                        st.markdown(f"<li style='font-size: 16px;'>{reason}</li>", unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)

    # Display the history log
    display_history_log()


def display_history_log():
    if 'history_log' in st.session_state:
        st.markdown("<h2 style='text-align: center;font-size: 28px; font-weight: bold;'>HISTORY LOG</h2>", unsafe_allow_html=True)
        history_log = st.session_state.history_log
        image_size = (300, 300)  # Set the desired image size
        # Add download button for the entire history log
        download_log_button_str = get_download_log_button_str(history_log)
        st.markdown(download_log_button_str, unsafe_allow_html=True)

        for i in range(0, len(history_log), 4):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                display_image(history_log[i], image_size)
            if i + 1 < len(history_log):
                with col2:
                    display_image(history_log[i + 1], image_size)
            if i + 2 < len(history_log):
                with col3:
                    display_image(history_log[i + 2], image_size)
            if i + 3 < len(history_log):
                with col4:
                    display_image(history_log[i + 3], image_size)

        

def display_image(history_item, image_size):
    image = history_item['image']
    image = image.resize(image_size)
    image_name = history_item['image_name']
    caption = history_item['caption']
    prediction_message = history_item['prediction_message']
    
    st.image(image, caption=f"{image_name}", use_column_width=True)
    st.markdown(f"<p style='text-align: center; font-size: 20px;'>{prediction_message}</p>", unsafe_allow_html=True)
    download_button_str = get_download_button_str(image, caption)
    st.markdown(download_button_str, unsafe_allow_html=True)

def update_history_log(history_item):
    if 'history_log' not in st.session_state:
        st.session_state.history_log = []

    new_entry = True
    for item in st.session_state.history_log:
        if(item['image_name'] == history_item['image_name']):
            new_entry = False
    
    if(new_entry):
        st.session_state.history_log.append(history_item)

def get_download_button_str(image, caption):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_str = base64.b64encode(buffered.getvalue()).decode()
    button_str = f'<a href="data:image/jpeg;base64,{image_str}" download="{caption}.jpg">Download Image</a>'
    return button_str

def get_download_log_button_str(history_log):
    zip_data = BytesIO()
    with zipfile.ZipFile(zip_data, 'w') as zf:
        for item in history_log:
            image = item['image']
            caption = item['caption']
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_str = base64.b64encode(buffered.getvalue()).decode()
            filename = f"{caption}.jpg"
            zf.writestr(filename, base64.b64decode(image_str))

    zip_data.seek(0)
    button_str = f'<a href="data:application/zip;base64,{base64.b64encode(zip_data.getvalue()).decode()}" download="history_log.zip">Download History Log</a>'
    return button_str
    

def show_help():

    st.write("<h2 style='text-align: center; font-weight:bold;'>About Image Authenticator</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px;'>This app uses a trained deep learning model and OpenCV analysis to classify whether an image is real or AI-generated.</p>"
                "<p style='font-size: 18px;'>You can upload an image using the 'Upload and Predict' tab and view the results in real-time.</p>"
                "<p style='font-size: 18px;'>The classification result is a weighted average of the deep learning model prediction and the OpenCV analysis.</p>"
                "<p style='font-size: 18px;'>The reasons for the classification are displayed, including the specific reasons detected by the OpenCV analysis.</p>",
                unsafe_allow_html=True)
    st.subheader("FAQ")

    # CSS code to increase font size and set container background
    css = '''
    <style>
       .stExpander [data-baseweb="accordion"] button [data-testid: 'stVerticalBlock']{
       font-size: 20px;
       background-color: #F2F4F6;
       }       

    </style>
    '''

    # Apply CSS code
    st.markdown(css, unsafe_allow_html=True)

    # Helper function to create an expander within a container
    def create_expander(title, content):
        with st.container():
            with st.expander(title, expanded=False):
                st.write(content)

    # Create expanders within containers
    create_expander("1: Are there any file size limitations for image uploads?",
                    "The application has the capability to process image files with a size of up to 200MB each.")

    create_expander("2: Can I download the images analyzed by the ImageAuthenticator?",
                    "Certainly! After the analysis process, you have the option to download the analyzed images. The downloaded files will retain their original names and include the predicted label, providing you with both the visual content and the corresponding classification for easy reference and further analysis.")

    create_expander("3: Can ImageAuthenticator detect all types of AI-generated images?",
                    "While the detector is effective at identifying common AI-generated images, it may not detect all types, especially if they are generated using advanced techniques.")

    create_expander("4: How does the ImageAuthenticator work?",
                    "The ImageAuthenticator uses a trained model to analyze images and determine whether they are real or AI-generated.")

    create_expander("5: What types of images can I upload to the application?",
                    "You can upload images in JPG, JPEG, or PNG format.")

    create_expander("6: How accurate is the AI-Generated Fake Image Detector?",
                    "The accuracy of the detector is around 90%. Generally, it strives to provide reliable results, but there may be instances of false positives or false negatives.")


if __name__ == '__main__':
    main()
