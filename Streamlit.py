import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page Title
st.title('Wine Quality Prediction App')

# Importing the dataset
@st.cache
def load_data():
    wine_dataset = pd.read_csv('winequality-red.csv')
    return wine_dataset

wine_dataset = load_data()

# Displaying the dataset
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(wine_dataset)

# Data Analysis and Visualization
st.subheader('Data Analysis and Visualization')

# Statistical summary of the dataset
st.write(wine_dataset.describe())

# Bar plot of quality counts
st.subheader('Quality Counts')
sns.catplot(x='quality', data=wine_dataset, kind='count', ax=plt.figure(figsize=(5,5)))

st.pyplot()

# Bar plot of volatile acidity vs Quality
st.subheader('Volatile Acidity vs Quality')
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
st.pyplot()

# Bar plot of citric acid vs Quality
st.subheader('Citric Acid vs Quality')
sns.barplot(x='quality', y='citric acid', data=wine_dataset)
st.pyplot()

# Correlation heatmap
st.subheader('Correlation Matrix')
correlation = wine_dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
st.pyplot()

# Model Building
st.subheader('Model Building')

# Separate features and target
X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Model training
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Model evaluation
st.subheader('Model Evaluation')
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
st.write('Accuracy:', accuracy)

# Prediction
st.subheader('Make a Prediction')

# Input form for user input
st.sidebar.title('Input Features')
fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.6, 16.0, 8.31)
volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.12, 1.58, 0.53)
citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.27)
residual_sugar = st.sidebar.slider('Residual Sugar', 0.9, 15.5, 2.5)
chlorides = st.sidebar.slider('Chlorides', 0.012, 0.611, 0.087)
free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1.0, 72.0, 15.0)
total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6.0, 289.0, 46.0)
density = st.sidebar.slider('Density', 0.990, 1.003, 0.9967)
pH = st.sidebar.slider('pH', 2.74, 4.01, 3.31)
sulphates = st.sidebar.slider('Sulphates', 0.33, 2.0, 0.64)
alcohol = st.sidebar.slider('Alcohol', 8.4, 14.9, 10.4)

# User input processing
input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

# Making prediction
prediction = model.predict(input_features)

# Displaying prediction
st.subheader('Prediction')
if prediction[0] == 1:
    st.write('This wine is predicted to be of good quality.')
else:
    st.write('This wine is predicted to be of bad quality.')

# Footer
st.markdown("""
    This app predicts the quality of red wine based on its physicochemical properties.
    Data source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
    """)

