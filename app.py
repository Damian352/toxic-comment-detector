import streamlit as st
from predict import predict

st.title("Toxic Comment Detector (PL)")

text = st.text_area("Wpisz komentarz")

if st.button("Analizuj"):
    probs = predict(text)

    st.subheader("Wyniki:")

    for i, val in enumerate(probs):
        st.write(f"Klasa {i}: {float(val):.3f}")