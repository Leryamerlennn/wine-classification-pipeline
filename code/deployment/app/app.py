import streamlit as st
import requests

st.title("Wine Classifier")
st.header("Input the wine features")

alcohol = st.number_input("Alcohol", min_value=11.0, max_value=14.8, value=13.0)
malic_acid = st.number_input("Malic Acid", min_value=0.74, max_value=5.80, value=2.34)
ash = st.number_input("Ash", min_value=1.36, max_value=3.23, value=2.36)
alcalinity_of_ash = st.number_input("Alcalinity of Ash", min_value=10.6, max_value=30.0, value=19.5)
magnesium = st.number_input("Magnesium", min_value=70.0, max_value=162.0, value=99.7)
total_phenols = st.number_input("Total Phenols", min_value=0.98, max_value=3.88, value=2.29)
flavanoids = st.number_input("Flavanoids", min_value=0.34, max_value=5.08, value=2.03)
nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", min_value=0.13, max_value=0.66, value=0.36)
proanthocyanins = st.number_input("Proanthocyanins", min_value=0.41, max_value=3.58, value=1.59)
color_intensity = st.number_input("Colour Intensity", min_value=1.3, max_value=13.0, value=5.1)
hue = st.number_input("Hue", min_value=0.48, max_value=1.71, value=0.96)
od280_od315_of_diluted_wines = st.number_input("OD280/OD315", min_value=1.27, max_value=4.00, value=2.61)
proline = st.number_input("Proline", min_value=278.0, max_value=1680.0, value=746.0)


features = [
    alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
    total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
    color_intensity, hue, od280_od315_of_diluted_wines, proline
]

if st.button("Classify Wine"):
    response = requests.post("http://api:8000/predict", json={"values": features})


    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Prediction: {prediction['prediction']}")
    else:
        st.write("Error: Could not get a prediction.")