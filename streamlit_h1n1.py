import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


Body_html = """

  <style>
  h1{
    color: #7a7477;

  }
   body{
    background-image: url(https://i.stack.imgur.com/HCfU2.png);
    #https://i.stack.imgur.com/9WYxT.png
    opacity: 0.9;
}


    <h1>How Likely One Takes H1N1 Vaccine?</h1>
</style>



"""
st.markdown(Body_html, unsafe_allow_html=True) #Body rendering

st.write(
"""

# How Likely One Takes H1N1 Vaccine?
"""
)

st.sidebar.header('User Imput Parameters')


st.sidebar.markdown("""
[Example CSV input file]("https://raw.githubusercontent.com/vanessa920/H1N1_seasonal_flu_shot_prediction/eda/Data/example_data.csv")
""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_feature():
        doctor_recc_h1n1 = st.sidebar.checkbox('H1N1 Vaccine Recommened by Doctor?')
        doctor_recc_seasonal = st.sidebar.checkbox('Seasonal Flu Vaccine Recommened by Doctor?')
        seasonal_vaccine = st.sidebar.checkbox('Received Seasonal Flu Vaccine?')
        health_worker = st.sidebar.checkbox('Working in Healthcare Industry?')
        h1n1_knowledge = st.sidebar.slider('Level of H1N1 Knowledge:', 0, 2, 1)
        h1n1_concern =st.sidebar.slider('Level of H1N1 Concern:', 0, 3, 2)
        opinion_h1n1_risk = st.sidebar.slider('Opinion: Is H1N1 Virus Risky?:', 0, 5, 1)
        opinion_h1n1_sick_from_vacc = st.sidebar.slider('How Risky do you think H1N1 Vaccine?:', 0, 5, 2)
        opinion_h1n1_vacc_effective = st.sidebar.slider('How Effective do you think H1N1 Vaccine?:', 0, 5, 3)
        opinion_seas_risk=st.sidebar.slider('How Much do you think Seasonal Flu is Risky?:', 0, 5, 5)
        opinion_seas_sick_from_vacc=st.sidebar.slider('How Risky do you think Seasonal Vaccine?:', 0, 5, 4)
        opinion_seas_vacc_effective=st.sidebar.slider('How Effective do you think Seasonal Vaccine?:', 0, 5, 3)
        data = {'doctor_recc_h1n1': doctor_recc_h1n1,
                'doctor_recc_seasonal': doctor_recc_seasonal,
                'seasonal_vaccine': seasonal_vaccine,
                'health_worker': health_worker,
                'h1n1_concern': h1n1_concern,
                'h1n1_knowledge': h1n1_knowledge,
                'opinion_h1n1_risk': opinion_h1n1_risk,
                'opinion_h1n1_sick_from_vacc': opinion_h1n1_sick_from_vacc,
                'opinion_h1n1_vacc_effective': opinion_h1n1_vacc_effective,
                'opinion_seas_risk': opinion_seas_risk,
                'opinion_seas_sick_from_vacc': opinion_seas_sick_from_vacc,
                'opinion_seas_vacc_effective': opinion_seas_vacc_effective}
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_feature()

    st.subheader('User Input Parameters')
    st.write(df)

    # df_drop = pd.read_pickle("Data/df_drop.pkl")
    # df_train=df_drop.values
    # X = df_train[:, 2:].astype(str)
    # y = df_train[:, 1].astype(str)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4444)
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoder.fit(X_train)
    # X_train = onehot_encoder.transform(X_train)
    # X_test = onehot_encoder.transform(X_test)
    # #ordinal encode target variable
    # label_encoder = LabelEncoder()
    # label_encoder.fit(y_train)
    # y_train = label_encoder.transform(y_train)
    # y_test = label_encoder.transform(y_test)
    # lr_model = LogisticRegression(solver="lbfgs")
    #lr_model.fit(X, y)

    lr_model=pd.read_pickle("Data/lr_model.pickle")
    prediction_proba=lr_model.predict_proba(df)
    st.write(prediction_proba[0][1])
