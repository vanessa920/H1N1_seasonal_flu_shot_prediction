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

  h5{
    color: lightgrey;
  }

 img{
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
    #width: 300px;
    height: auto;
    #opacity:0.9;
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

# Will a Person get the H1N1 Vaccine?

![vaccine_img](http://bento.cdn.pbs.org/hostedbento-prod/filer_public/spillover/images/viruses/definitions/vaccines.png)

#### Try the tool to predict whether a person will get the H1N1 vaccine using the information they shared about their healthcare background and opinions.

***

In Spring 2009, a pandemic caused by the H1N1 influenza virus ("Swine Flu"), swept across the world. A vaccine for the H1N1 flu virus became publicly available in October 2009.
In late 2009 and early 2010, the United States conducted the National 2009 H1N1 Flu Survey. More details about this dataset and features are available at [DrivenData.org](https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/).

***

"""
)


st.sidebar.markdown("""
[Example CSV input file]("https://raw.githubusercontent.com/vanessa920/H1N1_seasonal_flu_shot_prediction/eda/Data/example_data.csv")
""")

uploaded_file = st.sidebar.file_uploader("Upload your CSV input file", type=["csv"])
st.sidebar.subheader('Tool Input Parameters')
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_feature():
        doctor_recc_h1n1 = st.sidebar.checkbox('H1N1 Vaccine Recommended by Doctor?')
        doctor_recc_seasonal = st.sidebar.checkbox('Seasonal Flu Vaccine Recommended by Doctor?')
        seasonal_vaccine = st.sidebar.checkbox('Received Seasonal Flu Vaccine?')
        health_worker = st.sidebar.checkbox('Working in Healthcare Industry?')
        h1n1_knowledge = st.sidebar.slider('Level of H1N1 Knowledge:', 0, 2, 1)
        h1n1_concern =st.sidebar.slider('Level of H1N1 Concern:', 0, 3, 2)
        opinion_h1n1_risk = st.sidebar.slider('Opinion: Is H1N1 Virus Risky?:', 0, 5, 1)
        opinion_h1n1_sick_from_vacc = st.sidebar.slider('Opinion: H1N1 Vaccine Makes You Sick?:', 0, 5, 2)
        opinion_h1n1_vacc_effective = st.sidebar.slider('Opinion: Is H1N1 Vaccine Effective?:', 0, 5, 3)
        opinion_seas_risk=st.sidebar.slider('Opinion: Is Seasonal Flu Risky?:', 0, 5, 5)
        opinion_seas_sick_from_vacc=st.sidebar.slider('Opinion: Seasonal Vaccine Makes You Sick?:', 0, 5, 4)
        opinion_seas_vacc_effective=st.sidebar.slider('Opinion: Is Seasonal Vaccine Effective?:', 0, 5, 3)
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
    st.subheader('Based on the survey, the probability of getting the H1N1 Vaccine is:')
    st.header("{:.0%}".format(prediction_proba_lr[0][1]))  #print "{:.0%}".format(1/3)
    #st.write(prediction_proba_ada[0][1])
    # st.subheader('User Input Parameters')
    # st.write(df)
    # st.write("""
    # ![import_features]("Important Features/slide2.png")
    # """)
st.write(""" *** """)

st.write("""
Feature Coefficient in Logistic Model
""")
# feature_coef = pd.read_csv("Data/feature_coef.csv")
#
# ax = plt.bar(x='Coefficient', y='Feature', data=feature_coef,
#                  color = "#F3005E", order=feature_coef.sort_values('Coefficient', ascending = False).Feature)
#
# st.pyplot(feature_coef.Feature,use_container_width=True)

from PIL import Image
image_coef = Image.open('Important Features/Feature_Coeffient.jpg')
st.image(image_coef,use_column_width=True)

# image_rf = Image.open('Important Features/Features_in_rf.png')
# st.image(image_rf, caption='Feature Importance in Logistic Model',use_column_width=True)
st.write(""" *** """)
st.subheader("Methodology Diagram")
st.graphviz_chart('''
digraph {
    EDA -> Correlation
    EDA -> NAs
    EDA -> Transform
    Transform -> OneHotEncode
    Transform -> LabelEncode
    NAs -> MiceImput
    OneHotEncode -> Modeling
    MiceImput -> Modeling
    Modeling -> RandomForest
    Modeling -> CatBoost
    Modeling -> GBoost
    Modeling -> SVC
    Modeling -> DecisionTree
    Modeling -> NB
    Modeling -> KNN
    Modeling -> Logistic
    Logistic -> Pipeline
    ADA -> Pipeline
    ET -> Pipeline
    NB -> Pipeline
    KNN -> Pipeline
    SVC -> Pipeline
    RandomForest -> Pipeline
    RandomForest -> ImportantFeatures
    CatBoost -> ImportantFeatures
    ImportantFeatures -> Logistic
    Logistic -> Prediction
    Prediction -> Visualization
}
''')

st.markdown("""
##### Copyright@VanessaHu
""")
