import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained models
log_reg_model = joblib.load('C:/Users/potab/OneDrive - Solent University/COM 726 Dissertation/Dissertation 2024 folder/prediction-of-hospital-readmission-in-heart-failure-patients/COM-726/Logistic_Regression.joblib')
rf_model = joblib.load('C:/Users/potab/OneDrive - Solent University/COM 726 Dissertation/Dissertation 2024 folder/prediction-of-hospital-readmission-in-heart-failure-patients/COM-726/Random_Forest.joblib')
gb_model = joblib.load('C:/Users/potab/OneDrive - Solent University/COM 726 Dissertation/Dissertation 2024 folder/prediction-of-hospital-readmission-in-heart-failure-patients/COM-726/Gradient_Boosting.joblib')
nn_model = tf.keras.models.load_model('C:/Users/potab/OneDrive - Solent University/COM 726 Dissertation/Dissertation 2024 folder/prediction-of-hospital-readmission-in-heart-failure-patients/COM-726/Neural_Network.h5')

# Load the dataset to fit the scaler
file_path = 'C:/Users/potab/OneDrive - Solent University/COM 726 Dissertation/Dissertation 2024 folder/prediction-of-hospital-readmission-in-heart-failure-patients/COM-726/heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['DEATH_EVENT'])

# Define top features for each model
top_features_log_reg = ['age', 'serum_creatinine', 'ejection_fraction', 'time']
top_features_rf = ['age', 'serum_creatinine', 'ejection_fraction', 'time']
top_features_gb = ['age', 'serum_creatinine', 'ejection_fraction', 'time']
top_features_nn = ['age', 'serum_creatinine', 'ejection_fraction', 'time']

# Set up the app layout with sections
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "Visualization", "Prediction", "Information & Resources"])

# Home Section
if section == "Home":
    st.title("Heart Failure Readmission Prediction App")
    st.image("https://cdn.images.express.co.uk/img/dynamic/11/750x445/788234.jpg", caption='Heart Failure Prediction', use_column_width=True)
    st.write("""
    Welcome to the Heart Failure Readmission Prediction App. This tool is designed to help predict the likelihood of readmission 
    based on clinical data. Use the navigation menu on the left to explore different sections of the app, including data visualization, 
    prediction, and valuable resources on heart disease.
    """)

# Visualization Section
elif section == "Visualization":
    st.title("Data Visualization")
    st.write("Explore the dataset and visualize key features related to heart failure readmission.")

    # Sidebar for selecting visualizations
    st.sidebar.header("Select Visualization")
    viz_options = [
        "Age Distribution of Patients",
        "Serum Creatinine Levels by Death Event",
        "Kaplan-Meier Survival Curve",
        "Correlation Heatmap"
    ]
    selected_viz = st.sidebar.selectbox("Choose a visualization", viz_options)

    # Visualization 1: Age Distribution of Patients
    if selected_viz == "Age Distribution of Patients":
        st.subheader("Age Distribution of Patients")
        plt.figure(figsize=(10, 6))
        sns.histplot(data['age'], bins=20, kde=True, color='skyblue')
        plt.title('Age Distribution of Patients')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        st.pyplot(plt)

    # Visualization 2: Serum Creatinine Levels by Death Event
    elif selected_viz == "Serum Creatinine Levels by Death Event":
        st.subheader("Serum Creatinine Levels by Death Event")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='DEATH_EVENT', y='serum_creatinine', data=data, palette='Set2')
        plt.title('Serum Creatinine Levels by Death Event')
        plt.xlabel('Death Event')
        plt.ylabel('Serum Creatinine (mg/dL)')
        plt.xticks([0, 1], ['No', 'Yes'])
        st.pyplot(plt)

    # Visualization 3: Kaplan-Meier Survival Curve
    elif selected_viz == "Kaplan-Meier Survival Curve":
        st.subheader("Kaplan-Meier Survival Curve")
        from lifelines import KaplanMeierFitter

        kmf = KaplanMeierFitter()
        T = data['time']  # Duration of time
        E = data['DEATH_EVENT']  # Event occurred or censored

        kmf.fit(T, event_observed=E)

        plt.figure(figsize=(10, 6))
        kmf.plot()
        plt.title('Kaplan-Meier Survival Curve')
        plt.xlabel('Time (days)')
        plt.ylabel('Survival Probability')
        st.pyplot(plt)

    # Visualization 4: Correlation Heatmap
    elif selected_viz == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of the Heart Failure Clinical Dataset')
        st.pyplot(plt)



# Prediction Section
# Prediction Section
elif section == "Prediction":
    st.title("Heart Failure Readmission Prediction")

    # Sidebar for user input
    st.sidebar.header('User Input Parameters')
    
    def user_input_features():
        age = st.sidebar.slider('Age', min_value=0, max_value=120, value=60)
        anaemia = st.sidebar.selectbox('Anaemia', ('No', 'Yes'))
        creatinine_phosphokinase = st.sidebar.slider('Creatinine Phosphokinase (mcg/L)', min_value=0, max_value=8000, value=582)
        diabetes = st.sidebar.selectbox('Diabetes', ('No', 'Yes'))
        ejection_fraction = st.sidebar.slider('Ejection Fraction (%)', min_value=0, max_value=100, value=38)
        high_blood_pressure = st.sidebar.selectbox('High Blood Pressure', ('No', 'Yes'))
        platelets = st.sidebar.slider('Platelets (kiloplatelets/mL)', min_value=0, max_value=900000, value=265000)
        serum_creatinine = st.sidebar.slider('Serum Creatinine (mg/dL)', min_value=0.0, max_value=10.0, value=1.9)
        serum_sodium = st.sidebar.slider('Serum Sodium (mEq/L)', min_value=100, max_value=150, value=130)
        sex = st.sidebar.selectbox('Sex', ('Female', 'Male'))
        smoking = st.sidebar.selectbox('Smoking', ('No', 'Yes'))
        time = st.sidebar.slider('Follow-up Period (days)', min_value=0, max_value=300, value=4)

        data = {
            'age': age,
            'anaemia': 1 if anaemia == 'Yes' else 0,
            'creatinine_phosphokinase': creatinine_phosphokinase,
            'diabetes': 1 if diabetes == 'Yes' else 0,
            'ejection_fraction': ejection_fraction,
            'high_blood_pressure': 1 if high_blood_pressure == 'Yes' else 0,
            'platelets': platelets,
            'serum_creatinine': serum_creatinine,
            'serum_sodium': serum_sodium,
            'sex': 1 if sex == 'Male' else 0,
            'smoking': 1 if smoking == 'Yes' else 0,
            'time': time
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Display user input in an organized way
    st.subheader('User Input Parameters')
    st.write("Below are the parameters you selected:")
    st.table(input_df)

    # Model selection
    model_choice = st.sidebar.radio('Select Prediction Model', 
                                    ('Logistic Regression', 
                                     'Random Forest', 
                                     'Gradient Boosting', 
                                     'Neural Network'))

    # Select appropriate features based on the chosen model
    if model_choice == 'Logistic Regression':
        selected_features = top_features_log_reg
    elif model_choice == 'Random Forest':
        selected_features = top_features_rf
    elif model_choice == 'Gradient Boosting':
        selected_features = top_features_gb
    elif model_choice == 'Neural Network':
        selected_features = top_features_nn

    input_df_selected = input_df[selected_features]

    # Scale the input data
    scaler = StandardScaler()
    scaler.fit(X[selected_features])
    input_data_scaled = scaler.transform(input_df_selected)

    # Prediction
    if model_choice == 'Logistic Regression':
        pred = log_reg_model.predict(input_data_scaled)
    elif model_choice == 'Random Forest':
        pred = rf_model.predict(input_data_scaled)
    elif model_choice == 'Gradient Boosting':
        pred = gb_model.predict(input_data_scaled)
    elif model_choice == 'Neural Network':
        pred = (nn_model.predict(input_data_scaled) > 0.5).astype("int32")

    result = int(pred[0]) if model_choice != 'Neural Network' else int(pred[0][0])

    # Display prediction result with more context and visual feedback
    st.subheader('Prediction Result')
    if result == 1:
        st.error("Prediction: The patient is likely to be readmitted.")
    else:
        st.success("Prediction: The patient is not likely to be readmitted.")

    st.markdown(
        """
        **Model Used:** {}
        """.format(model_choice)
    )


# Information & Resources Section
elif section == "Information & Resources":
    st.title("Information & Resources")
    st.write("""
    Heart disease is a leading cause of death globally. It's important to be informed about the risks, symptoms, and treatment options.
    Below are some valuable resources to learn more about heart disease:
    """)

    st.markdown("""
    - [World Health Organization (WHO) - Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases#tab=tab_1)
    - [American Heart Association](https://www.heart.org/)
    - [British Heart Foundation](https://www.bhf.org.uk/)
    - [Centers for Disease Control and Prevention (CDC) - Heart Disease](https://www.cdc.gov/heartdisease/index.htm)
    """)

    st.write("""
    Regular exercise, a balanced diet, and avoiding smoking can significantly reduce your risk of heart disease. 
    If you have any concerns, consult a healthcare professional.
    """)
