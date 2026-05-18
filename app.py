import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Title
st.title("Logistic Regression Prediction")

# Dataset
data = {
    "study_hours": [1,2,3,4,5,6,7,8,9,10],
    "pass": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["study_hours"]]
y = df["pass"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# User Input
hours = st.number_input("Enter Study Hours", 0.0, 15.0)

# Prediction
if st.button("Predict"):

    prediction = model.predict([[hours]])[0]
    probability = model.predict_proba([[hours]])[0][1]

    st.write("Pass Probability:", round(probability, 2))

    if prediction == 1:
        st.success("Student Will Pass")
    else:
        st.error("Student Will Fail")