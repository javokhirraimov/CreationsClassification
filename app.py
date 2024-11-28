import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
from platform import system

system_type = platform.system()
if system_type == "Linux": pathlib.WindowsPath = pathlib.PosixPath


#title
st.title("DONIYORBEK TUGULGAN KUNINIZ BILAN TABRIKLIMAN ")
st.text("Assalomu aleykum Doniyor aka tuģulgan kuniz bilan tabrikliman umriz uzoq bòsin taniz soģ bòsin bunaqa yoshlani kòpini korib yuring🥳")
#rasmni joylash
file = st.file_uploader('Image load')
if file:
    st.image(file)
    #PIL convert

    img = PILImage.create(file)

    #model
    model = load_learner("creature_model.pkl")

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")

    #barchart
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
