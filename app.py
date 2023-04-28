import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import torch
import json
import os

import src.engine as eng


# load from the path
model_path = os.path.abspath("weights/Quantized_ResNet_21.pt")
model = torch.load(model_path)

# json file to get classes
with open('int_to_class.json', 'r') as f:
    labels = json.load(f)
species_names = labels.values()


#st.set_page_config(
 #   page_title="What's that bird?",
  #  page_icon=":bird:",
   # #layout="wide",
    #initial_sidebar_state="expanded",)

# display page text
image_path = os.path.abspath("media/peacock1.jpg")
image = Image.open(image_path)
st.title('Know The Bird')
Info = """
    India has more than 1250 bird species. This web app classifies only 25 species.
    More species will be added.

    """
st.write(Info)
st.image(image, caption='A Peacock')

Info = """
    You can select one image from the sibebar select option to predict the bird's species. 
    You can also upload any image from your system to know the species of the bird.
    """
st.write(Info)

st.markdown(
    """
    <h4 style="text-align: left; color: black;">
        <span style="font-size: larger;">🐦</span>
        [Once the image is selected click "Predict" and you will have the answer]
        <span style="font-size: larger;">🐦</span>
    </h4>
    make sure to unselect the image before using either of the options
    
    """,
    unsafe_allow_html=True
)

# make a dict to allow users to select image
image_dict = {
    "Image 1": os.path.abspath("media/ML315345341.jpg"),
    "Image 2": os.path.abspath("media/ML390859511.jpg"),
    "Image 3": os.path.abspath("media/ML515331161.jpg"),
    "Image 4": os.path.abspath("media/Indian_pitta_bd.jpg"),
    "Image 5": os.path.abspath("media/kingfisher-1.jpg")
    }

select_image = st.sidebar.selectbox(
    "Select an image", [""]+list(image_dict.keys()))


st.sidebar.write("### 25 Indian Bird Species used are:")
for name in species_names:
    st.sidebar.write(f"-{name}")

if select_image:
    image_path = image_dict[select_image]
    image = Image.open(image_path)
    st.image(image, caption=select_image)  

# allow users to upload their own prefered image
img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
if img_file:
    img = Image.open(img_file)
    st.image(img, caption="Uploaded image")   

# Make prediction on selected or uploaded image
if select_image or img_file:
    if st.button("Predict"):
        if select_image:
            result = eng.predict_image(image_path, model, labels)
        else:
            save_image_path = os.path.join("media", img_file.name)
            img.save(save_image_path)
            result = eng.predict_image(save_image_path, model, labels)
        st.success("Predicted bird is: " + result)
        st.balloons()



