import streamlit as st
import pickle
import numpy as np


# Load files
model = pickle.load(open("model.pkl", "rb"))

symptoms = pickle.load(open("symptoms.pkl", "rb"))


try:
    encoder = pickle.load(open("label_encoder.pkl", "rb"))
except:
    encoder = None


# App title
st.title("Disease Prediction System")

st.write("Select your symptoms to predict the disease")


# User input
selected_symptoms = st.multiselect(
    "Choose Symptoms",
    symptoms
)


# Prediction button
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")

    else:

        # Create input vector
        input_data = np.zeros(len(symptoms))

        for symptom in selected_symptoms:
            index = symptoms.index(symptom)
            input_data[index] = 1


        # Reshape for model
        input_data = input_data.reshape(1, -1)


        # Prediction
        prediction = model.predict(input_data)


        # Convert label back
        if encoder:
            disease = encoder.inverse_transform(prediction)[0]
        else:
            disease = disease = encoder.inverse_transform(prediction)[0]


        st.success(f"Predicted Disease: {disease}")
