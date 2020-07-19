import tensorflow as tf
import streamlit as st

best_model = tf.keras.models.load_model('model/')
best_model.summary()

st.write("""
         # Echocardiogram View Prediction
         """
         )
st.write("This web app classify the view of Echocardiogram")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, best_model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = best_model.predict(img_reshape)
        return prediction
if file is None:
        st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, best_model)
    
    if np.argmax(prediction) == 0:
        st.write("AP2")
    elif np.argmax(prediction) == 1:
        st.write("AP3")
    elif np.argmax(prediction) == 2:
        st.write("AP4")
    elif np.argmax(prediction) == 3:
        st.write("AP5")
    elif np.argmax(prediction) == 4:
        st.write("PLAX")
    elif np.argmax(prediction) == 5:
        st.write("PSAX-AP")
    elif np.argmax(prediction) == 6:
        st.write("PSAX-AV")
    elif np.argmax(prediction) == 7:
        st.write("PSAX-MID")
    
    else:
        st.write("PSAX-MV")
    
    st.text("Probability (0: AP2, 1: AP3, 2: AP4, 3: AP5, 4: PLAX, 5: PSAX-AP, 6: PSAX-AV, 7: PSAX-MID, 8: PSAX-MV")
    st.write(prediction)
