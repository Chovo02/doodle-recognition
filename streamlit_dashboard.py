from src.model.LeNet5 import LeNet5
from src.model.AlexNet import AlexNet
from src.model.VGGNet import VGGNet
from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
st.set_page_config(page_title="Doodle Recognition", layout="wide")


def import_model():
    lenet5 = LeNet5()
    lenet5.build()
    lenet5.compile()
    lenet5.load_weights(path="./data/weights/LeNet.h5")

    alexnet = AlexNet()
    alexnet.build()
    alexnet.compile()
    alexnet.load_weights(path="./data/weights/AlexNet.h5")

    vggnet = VGGNet()
    alexnet.build()
    alexnet.compile()
    vggnet.load_weights(path="./data/weights/VGGNet.h5")
    return lenet5, alexnet, vggnet


lenet5, alexnet, vggnet = import_model()


st.title("Doodle Recognition")
col1, col2 = st.columns(2)
with col1:
    image_data = st_canvas(width=500, height=500, key="canvas", fill_color="#000", stroke_color="#fff")
with col2:
    st.header("Prediction")    
    text = st.empty()


with st.sidebar:
    model = st.selectbox("Model", ["LeNet5", "AlexNet", "VGGNet"])

if image_data is not None: 
    try:
        img = image_data.image_data.astype(np.uint8)
    except AttributeError:
        exit()
    img = Image.fromarray(img).convert("L")
    img = img.resize((28, 28))
    img.save("./data/test.png")
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    
    if img.max() == 0:
        text.write("No image detected")
        exit()
    
    if model == "LeNet5":
        top, top_encoded = lenet5.predict(img)
    elif model == "AlexNet":
        top, top_encoded = alexnet.predict(img)
    else:
        top, top_encoded = vggnet.predict(img)
    
    text.markdown(f"""
               1. {top_encoded[0]} ({round(top[0]*100, 2)} %)\n 
               2. {top_encoded[1]} ({round(top[1]*100, 2)} %)\n 
               3. {top_encoded[2]} ({round(top[2]*100, 2)} %)\n 
               2. {top_encoded[3]} ({round(top[3]*100, 2)} %)\n 
               5. {top_encoded[4]} ({round(top[4]*100, 2)} %)""")