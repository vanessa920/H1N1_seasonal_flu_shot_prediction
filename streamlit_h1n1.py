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
  h2{
    color:#F3005E;
    margin: 0;
    position: absolute;
    top: 50%;
    left: 50%;
    margin-right: -50%;
    transform: translate(-50%, -50%)
  }
  h3{
    color:#898989;
    display: block;
    margin-left: auto;
    margin-right: auto;
    size: 200%;
  }

  h4{
    color: #7a7477;
  }

 img{
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
    #width: 300px;
    height: auto;
    opacity:0.8;
}

   body{
    background-image: url(https://i.stack.imgur.com/HCfU2.png);
    #https://i.stack.imgur.com/9WYxT.png
    #opacity: 0.5;
}

</style>



"""
st.markdown(Body_html, unsafe_allow_html=True) #Body rendering

st.write(
"""

# How Likely One Takes H1N1 Vaccine?

![vaccine_img](http://bento.cdn.pbs.org/hostedbento-prod/filer_public/spillover/images/viruses/definitions/vaccines.png)

#### Try the tool to predict whether people got H1N1 vaccines using information they shared about their healthcare backgrounds and opinions.

***

In Spring 2009, a pandemic caused by the H1N1("Swine Flu") influenza virus, swept across the world. A vaccine for the H1N1 flu virus became publicly available in October 2009.
In late 2009 and early 2010, the United States conducted the National 2009 H1N1 Flu Survey. More details about this dataset and features available at [DrivenData](https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/).

***

"""
)

#st.sidebar.header('User Imput Parameters')


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
    #ada_model=pd.read_pickle("models/ada_model.pickle")
    prediction_proba_lr=lr_model.predict_proba(df)
    #prediction_proba_ada=ada_model.predict_proba(df)
    st.subheader('Based on survey, the possibility of One Receives H1N1 Vaccine is:')
    st.header("{:.0%}".format(prediction_proba_lr[0][1]))  #print "{:.0%}".format(1/3)
    #st.write(prediction_proba_ada[0][1])
    # st.subheader('User Input Parameters')
    # st.write(df)
    st.write("""
    ![import_features_logistic_Regression]("Important Features/slide2.png")
    """)
