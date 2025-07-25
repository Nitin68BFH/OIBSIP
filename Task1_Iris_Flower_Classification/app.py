import streamlit as st
import numpy as np
import os
import joblib

# This always finds the model relative to the location of app.py, wherever the app is run
model_path = os.path.join(os.path.dirname(__file__), 'iris_model.pkl')
model = joblib.load(model_path)


#loading model
model = joblib.load('iris_model.joblib')
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
#user input function
def user_input():
    sepal_length = st.slider('Sepal Length (cm)' , 4.0,8.0,5.1)
    sepal_width = st.slider('Sepal Width (cm)',2.0,4.5,3.5)

    petal_length = st.slider('Petal Length (cm)',1.0,7.0,1.4)
    petal_width = st.slider('Petal Width (cm)',0.1,2.5,0.2)

    return np.array([[sepal_length , sepal_width , petal_length , petal_width]])

#prediction function
def predict_flower(data):
    prediction = model.predict(data)[0]
    return prediction

# function for steamlit app
def main():
    st.title('🌸Iris Flower Prediction')

    input_data = user_input()

    if st.button('Predict'):
        pred_class = predict_flower(input_data) 
        st.success(f'Predicted Iris Class :{class_names[pred_class]}')

if __name__ == '__main__':
    main()