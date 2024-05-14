import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the model from the pickle file
loaded_dict = pickle.load(open("./model/churn-scaler.pickle", "rb"))
loaded_model = loaded_dict['model']  # Access the model from the dictionary
loaded_scaler = loaded_dict['scaler']  # Access the scaler from the dictionary
# Define the prediction function
def ValuePredictor(to_predict_list):
    # to_predict = np.array(to_predict_list).reshape(1, 19)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # cat_cols_ohe = ['PaymentMethod', 'Contract', 'InternetService']
    to_predict_list[num_cols] = loaded_scaler.transform(to_predict_list[num_cols])
    result = loaded_model.predict(to_predict_list)
    return result[0]

def convert_form_values_to_numerical(gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges):
    # Define the mappings for categorical variables
    gender_mapping = {'Female': 0, 'Male': 1}
    senior_citizen_mapping = {'No': 0, 'Yes': 1}
    partner_mapping = {'No': 0, 'Yes': 1}
    dependents_mapping = {'No': 0, 'Yes': 1}
    phone_service_mapping = {'No': 0, 'Yes': 1}
    multiple_lines_mapping = {'No phone service': 0, 'No': 1, 'Yes': 2}
    internet_service_mapping = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    online_security_mapping = {'No': 0, 'No internet service': 1, 'Yes': 2}
    online_backup_mapping = {'No': 0, 'No internet service': 1, 'Yes': 2}
    device_protection_mapping = {'No': 0, 'No internet service': 1, 'Yes': 2}
    tech_support_mapping = {'No': 0, 'No internet service': 1, 'Yes': 2}
    streaming_tv_mapping = {'No': 0, 'No internet service': 1, 'Yes': 2}
    streaming_movies_mapping = {'No': 0, 'No internet service': 1, 'Yes': 2}
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    paperless_billing_mapping = {'No': 0, 'Yes': 1}
    payment_method_mapping = {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3}

    # Convert the form values to numerical values
    gender = gender_mapping[gender]
    senior_citizen = senior_citizen_mapping[senior_citizen]
    partner = partner_mapping[partner]
    dependents = dependents_mapping[dependents]
    tenure = float(tenure)
    phone_service = phone_service_mapping[phone_service]
    multiple_lines = multiple_lines_mapping[multiple_lines]
    internet_service = internet_service_mapping[internet_service]
    online_security = online_security_mapping[online_security]
    online_backup = online_backup_mapping[online_backup]
    device_protection = device_protection_mapping[device_protection]
    tech_support = tech_support_mapping[tech_support]
    streaming_tv = streaming_tv_mapping[streaming_tv]
    streaming_movies = streaming_movies_mapping[streaming_movies]
    contract = contract_mapping[contract]
    paperless_billing = paperless_billing_mapping[paperless_billing]
    payment_method = payment_method_mapping[payment_method]
    monthly_charges = float(monthly_charges)
    total_charges = float(total_charges)

    # Return the numerical values as a list
    return [gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges]

# Define the Streamlit app
def run():
    st.title("Churn Customer Indihome")
    st.write("Melakukan Prediksi Churn Customer Indihome dengan menggunakan model Logistic Regression")
    st.sidebar.title("Tugas ")

    menu = st.sidebar.selectbox('Menu', ['Customer Churn'])
    if menu == 'Customer Churn':
        col1, col2 = st.columns(2)
        # Create inputs for all the features
        gender = col1.selectbox('Gender', ['Female', 'Male'])
        senior_citizen = col1.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = col1.selectbox('Partner', ['No', 'Yes'])
        dependents = col1.selectbox('Dependents', ['No', 'Yes'])
        tenure = col1.number_input('Tenure', min_value=0.0, step=1.0)
        phone_service = col1.selectbox('Phone Service', ['No', 'Yes'])
        multiple_lines = col1.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
        internet_service = col1.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = col1.selectbox('Online Security', ['No', 'No internet service', 'Yes'])
        online_backup = col1.selectbox('Online Backup', ['No', 'No internet service', 'Yes'])
        device_protection = col2.selectbox('Device Protection', ['No', 'No internet service', 'Yes'])
        tech_support = col2.selectbox('Tech Support', ['No', 'No internet service', 'Yes'])
        streaming_tv = col2.selectbox('Streaming TV', ['No', 'No internet service', 'Yes'])
        streaming_movies = col2.selectbox('Streaming Movies', ['No', 'No internet service', 'Yes'])
        contract = col2.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = col2.selectbox('Paperless Billing', ['No', 'Yes'])
        payment_method = col2.selectbox('Payment Method', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
        monthly_charges = col2.number_input('Monthly Charges', min_value=0.0, step=0.1)
        total_charges = col2.number_input('Total Charges', min_value=0.0, step=0.1)
    # When the 'Predict' button is pressed, make the prediction and display it
        subcol1, subcol2 = col2.columns(2)
        if subcol1.button("Predict"):
            if not all([gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges]):
                subcol2.markdown('**⚠️ Please fill in all inputs.**')
            else:
                to_predict_list = convert_form_values_to_numerical(gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges)
                print(to_predict_list)
                feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
                                'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                                'MonthlyCharges', 'TotalCharges']
                to_predict_df = pd.DataFrame([to_predict_list], columns=feature_names)
                print(to_predict_df)
                print(to_predict_df.info())
                result = ValuePredictor(to_predict_df)
                print(result)
                if int(result) == 1:
                    prediction = 'The Customer is Churn'
                else:
                    prediction = 'The Customer is No Churn'
                subcol2.markdown(prediction)
if __name__ == '__main__':
    run()