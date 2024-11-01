import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from PIL import Image

st.title('Titanic Futurista')

def load_data():
    data = pd.read_csv(r"train (2).csv")
    return data

def load_model():
    loaded_model = joblib.load("knn.joblib")
    return loaded_model

data = load_data()
st.write(data.head())

st.bar_chart(data.head(20).Age)



st.title('TITANIC APP')
st.text("Inputs para modelo")

RommService = st.number_input("Rommservice")
FoodCourt = st.number_input("FoodCourt")
ShoppingMall = st.number_input("ShoppingMall")
Spa= st.number_input("Spa")

clicked = st.button("Enviar datos")
model = load_model()
if clicked:
    print("modelo procesando")
    resultado = model.predict(pd.DataFrame({"RommService": [RommService],
             "FoodCourt":[FoodCourt],
             "ShoppingMall": [ShoppingMall],
             "Spa":[Spa]}))
    st.text("El resultado del modelo es: {}".format(resultado))
 

st.text("Ubicación en kaggle")

image = Image.open("Captura de pantalla 2023-05-15 192103.png")

st.image(image)

arr = [data.Spa]
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)

