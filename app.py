import streamlit as st

st.title("Smart MCQ Solver")

question = st.text_area("Enter your question")

if st.button("Solve"):
    st.write("Model deployment is working!")
