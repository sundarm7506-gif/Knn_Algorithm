
from operator import le
import streamlit as st 
import numpy as np 
import pickle 
 
with open("knn_model.pkl","rb") as f:
    model=pickle.load(f)
    
st.title("KNN Classification App")
st.write("Predict Class using Random Dataset")

f1=st.number_input("Feature 1", 1, 100, 50)
f2=st.number_input("Feature 2", 1, 100, 50)
f3=st.number_input("Feature 3", 1, 100, 50)
 
if st.button("predict"):
    input_data=np.array([[f1, f2, f3]])
    prediction=model.predict(input_data)
    class_name = le.inverse_transform(prediction)
    st.success(f"Predicted Class: **{class_name[0]}**")
    

    
        