'''
Iris Flower Classfier using simple machine learning
@author: Suyash Shivaji Phatak
Date:17/06/2020
'''

# Importing libraries
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Creaing Heading and small description
st.write("""
# Simple Iris Flower Prediction App
## *Created by* **Suyash Shivaji Phatak**

This app predics the **iris** flower type:
""")

# Creating a sidebar for getting input parameters
st.sidebar.header('User Input Parameters')

# function for creating slide bar for input
def user_input_style():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'Sepal Length': sepal_length,
            'Sepal Width': sepal_width,
            'Petal Length': petal_length,
            'Petal Width': petal_width}
    style = pd.DataFrame(data, index=[0])
    return style

# Creating Variable for storing the fuction value
df = user_input_style()

# Creating markdown and adding data frame input
st.subheader('User Input Parameters')
st.write(df)

# Calling iris dataset feom sklearn
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# classifier class
clf = RandomForestClassifier()
clf.fit(X, Y)

# Making Prediction and its probability
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Creating tables and uploading the datasets to the tables
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
