import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import torch
import json
import os

import src.engine as eng
from main import create_model

# st.cache_resource to keep memory usage in limit
@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes=25)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    check = torch.load(model_path, map_location=device)
    model.load_state_dict(check['model_state_dict'])
    optimizer.load_state_dict(check['optimizer_state_dict'])
    epoch = check['epoch']
    loss = check['best_loss']

    return model

# load from the path
model_path = os.path.abspath("weights/ResNet_21.pt")
model = load_model(model_path)

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
st.title('Indian Bird Classification')
Info = """
    You can select one image from the sibebar select option to predict the bird's species. 
    You can also upload any image from your system to classify the species of the bird.

    We will be updating more features like fun facts and other utilities like map and migrations strategy.
    """
st.write(Info)
st.image(image, caption='The way of Jungle')

st.markdown(
    """
    <h4 style="text-align: left; color: violet;">
        <span style="font-size: larger;">🐦</span>
        [Once the image is selected click "Predict" and you will have the answer]
        <span style="font-size: larger;">🐦</span>
    </h4>
    """,
    unsafe_allow_html=True
)

# make a dict to allow users to select image
image_dict = {
    "Image 1": os.path.abspath("image_data/test/Brown-Headed Barbet/ML77872441.jpg"),
    "Image 2": os.path.abspath("image_data/test/Hoopoe/ML108410391.jpg"),
    "Image 3": os.path.abspath("image_data/test/Ruddy Shelduck/ML206598971.jpg"),
    "Image 4": os.path.abspath("image_data/test/Jungle Babbler/ML125506401.jpg")
}
select_image = st.sidebar.selectbox(
    "Select an image", [""]+list(image_dict.keys()))


st.sidebar.write("### 25 Indian Bird Species this used are:")
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
            save_image_path = os.path.join("upload_images", img_file.name)
            image.save(save_image_path)
            result = eng.predict_image(save_image_path, model, labels)
        st.success("Predicted bird species is: " + result)
        st.balloons()



