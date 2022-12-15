import time
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sklearn
#import SessionState

#session = SessionState.get(run_id=0)
def clear_form():
    st.session_state["Pregnancies"] = 0
    st.session_state["Glucose"] = 0
    st.session_state["BloodPressure"] = 0
    st.session_state["SkinThickness"] = 0
    st.session_state["Insulin"] = 0
    st.session_state["BMI"] = 0
    st.session_state["PedigreeFunction"] = 0
    st.session_state["Age"] = 0
    

def do_predict(input_data, scaler):
    """get probabilities of each label"""
    user_input = np.array([input_data]).astype(np.float64)
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)

    # st.write("prediction: " + str(prediction))

    return int(prediction), prediction_proba

def main():
    st.set_page_config(page_title="Diabetes Detection", page_icon=":hospital:")

    st.markdown("<h2 style='text-align: center; color: black;'>Diagnosing Diabetes</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>By Using the Pima Indian Database</h4><br/>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: black;'>Please fill the form below and click the predict button to see how likely your patient has diabetes.</h5>", unsafe_allow_html=True)

    st.markdown("<br/><hr>", unsafe_allow_html=True)

    # Here is the list of features to be sent to the model
    # st.write(model.features)

    inputs = []
    
    c1, c2 = st.columns([1,1])

    predicted = False

    with c1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, help="")
        Glucose = st.number_input("Glucose", min_value=0, max_value=400, help="")
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, help="")
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=200, help="")
        

    with c2:
        Insulin = st.number_input("Insulin", min_value=0, max_value=1000, help="")
        BMI = st.number_input("BMI", min_value=0, max_value=99, help="")
        PedigreeFunction = st.number_input("Pedigree Function", min_value=0, max_value=99, help="")
        Age = st.number_input("Age", min_value=0, max_value=99, help="")

    st.markdown("<hr>", unsafe_allow_html=True)

    c3, c4, _ = st.columns([1,1, 6])
    with c3:
        if st.button("Predict"):
            st.spinner()

            inputs.append(Pregnancies)
            inputs.append(Glucose)
            inputs.append(BloodPressure)
            inputs.append(SkinThickness)
            inputs.append(Insulin)
            inputs.append(BMI)
            inputs.append(PedigreeFunction)
            inputs.append(Age)

            predicted_class, probas = do_predict(inputs, scaler)
            # st.write(predicted_class)
            # st.write(probas)
            probas = [x * 100 for x in probas]
            predicted = True

    with c4:
        if st.button("Clear"):
            clear_form()
            # session.run_id += 1
            

    if predicted:

        st.markdown("<br/>", unsafe_allow_html=True)

        probabilityOfDiabetes = round(probas[0][1], 2)

        if probabilityOfDiabetes < 50:
            st.info('It is **unlikely** that your patient has diabetes. Because, my calculations show **{}%** probability.'.format(str(probabilityOfDiabetes)))
        elif probabilityOfDiabetes <= 75 and probabilityOfDiabetes >= 50:
            st.warning('Sorry, I must stay neutral. Your patient may or may not have diabetes. Because, my calculations show **{}%** probability.'.format(str(probabilityOfDiabetes)))
        elif probabilityOfDiabetes <= 90 and probabilityOfDiabetes > 75:
            st.success('It is **likely** that your patient has diabetes. Because, my calculations show **{}%** probability.'.format(str(probabilityOfDiabetes)))
        else:
            st.success('It is **very likely** that your patient has diabetes. Because, my calculations show **{}%** probability.'.format(str(probabilityOfDiabetes)))

        st.error('By the way, do not forget that I am just a prototype! Don\'t take my word for it.')

if __name__ == '__main__':
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    main()
