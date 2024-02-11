import streamlit as sl
import pickle
import numpy as np

# if 'value' not in sl.session_state:
#     sl.session_state['value'] = 0

# if sl.button("Tambah"):
#     sl.session_state['value'] += 1

# sl.write(f"Counter: {sl.session_state['value']}")

# write the title
sl.title("Coba coba yaa ehehe")

# load the model
if 'model' not in sl.session_state:
    model = pickle.load(open('model.sav', 'rb'))
    sl.session_state['model'] = model


# coz the model has 3 features, we will create 

# """
#     Because the model has 3 features,
#     we will create user input that matches the type of features. 
#     Because the three features are numbers, user input is created using input_number
# """

area_input = sl.number_input("Insert Area of the House :house:")
bedroom_input = sl.number_input("Insert How Many Bedrooms of the House :bedroom:")
age_input = sl.number_input("Insert Age of the House :age:")

if sl.button("Model Predict"):
    data = np.array([area_input, bedroom_input, age_input]).reshape(1,-1)
    prediction = sl.session_state['model'].predict(data)
    sl.write(f'Prediction Result: $ {np.round(prediction[0], 2)}')
else:
    sl.write("Please input the feature above to start modelling")