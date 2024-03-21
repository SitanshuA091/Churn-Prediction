import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
model = pickle.load(open("model.sav", 'rb')


from Preprocessing import preprocess 


def main():
    st.title('Customer Churn Prediction App(Sitanshu Anmol)')
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional telecommunication use case.
    This Web application is functional for both online prediction and batch data prediction. n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    image = Image.open('App.jpg')
    add_selectbox = st.sidebar.selectbox(
    "Prediction Type", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        st.subheader("User data")
        Partner = st.selectbox('Partner:',('Yes',"No"))
        Gender = st.selectbox('Gender:',('Male','Female'))
        seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
        dependents = st.selectbox('Dependent:', ('Yes', 'No'))
        st.subheader("Payment data")
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        PaymentMethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
        st.subheader("Contract Data")
        contract = st.selectbox('Contract Type',('Month-to-month','One year', 'Two Year'))

        st.subheader("Services signed up for")
        mutliplelines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
        phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
        internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
        onlinebackup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
        deviceprotection = st.selectbox("Does the customer have deviceprotection",('Yes','No'))
        techsupport = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))
        streamingtv = st.selectbox("Does the customer stream TV", ('Yes','No','No internet service'))
        streamingmovies = st.selectbox("Does the customer stream movies", ('Yes','No','No internet service'))

        data = {
                'Gender':Gender,
                'Senior Citizen': seniorcitizen,
                'Partner': Partner,
                'Dependents': dependents,
                'Tenure Months':tenure,
                'Phone Service': phoneservice,
                'Multiple Lines': mutliplelines,
                'Internet Service': internetservice,
                'Online Security': onlinesecurity,
                'Online Backup': onlinebackup,
                'Device Protection':deviceprotection,
                'Tech Support': techsupport,
                'Streaming TV': streamingtv,
                'Streaming Movies': streamingmovies,
                'Contract': contract,
                'Paperless Billing': paperlessbilling,
                'Payment Method':PaymentMethod,
                'Monthly Charges': monthlycharges,
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)
        preprocess_df = preprocess(features_df)

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will exit the services of the organization.')
            else:
                st.success('No, the customer is willing to continue with the Services.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data)
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the customer will terminate the service.',
                                                    0:'No, the customer is willing to continue with the Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
        main()
