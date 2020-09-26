import streamlit as st
import pandas as pd
from joblib import load
import sklearn 

st.title("Heart Failure Prediction")
st.write("""
*by Anh*

# Introduction
Hi, my name is Phan Anh, a Vietnamese high school student,
And this is a simple project I spun up on the Heart Failure dataset on Kaggle.
In [the original notebook](https://www.kaggle.com/duongphananh/heart-failure-prediction/output), 
I did a rather deep dive on data analysis,
however, in this webapp which I zapped up in one night, 
I'll be only attempting to demonstrate the model. Here goes,

\n\n

# Prediction
""")


raw = pd.read_csv("heart_failure_clinical_records_dataset.csv")
dataframe = raw.head().drop("DEATH_EVENT", axis=1)

st.sidebar.write("# Build-a-person")
st.sidebar.write("## Boolean Data")
dataframe["sex"] = st.sidebar.selectbox("Choose person gender:", ["Male", "Female"])
dataframe["smoking"] = st.sidebar.checkbox("Smoker")
dataframe["anaemia"] = st.sidebar.checkbox("Anaemia")
dataframe["diabetes"] = st.sidebar.checkbox("Diabetes")
dataframe["high_blood_pressure"] = st.sidebar.checkbox("High blood pressure")


st.sidebar.write("## Numeric Data")
dataframe["age"] = st.sidebar.slider("Age", min_value=40, max_value=95, value=60, step=1)
dataframe["creatinine_phosphokinase"] = st.sidebar.slider("Creatinine Phosphokinase (mcg/L)", min_value=23, max_value=7861, value=250, step=1)
dataframe["ejection_fraction"] = st.sidebar.slider("Ejection Fraction (%)", min_value=14, max_value=80, value=38, step=1)
dataframe["platelets"] = st.sidebar.slider("Platelets (kiloplatelets/mL)", min_value=25100, max_value=850000, value=262000, step=100)
dataframe["serum_creatinine"] = st.sidebar.slider("Serum Creatinine (mg/dL)", min_value=0.5, max_value=9.4, value=1.1, step=0.1)
dataframe["serum_sodium"] = st.sidebar.slider("Serum Sodium (mEq/L)", min_value=113, max_value=148, value=137, step=1)
dataframe["time"] = st.sidebar.slider("Follow-up period (days)", min_value=4, max_value=285, value=115, step=1)
dataframe["sex"] = 1 if dataframe["sex"].iloc[0] == "Male" else 0


st.write("## Normal Dataframe")
st.write("""
*Note: There are five rows in this dataframe,
but they all points to the user input.
The reason why I am doing as such is because
dataframes with 1 row is very hard to see in the Firefox broswer.*
""")
st.dataframe(dataframe)


normalize_cols = ["ejection_fraction","serum_creatinine", "serum_sodium", "time", "creatinine_phosphokinase", "platelets"]
normalized_df = dataframe
normalized_df[normalize_cols] = ((normalized_df[normalize_cols] - raw[normalize_cols].min()) / (raw[normalize_cols].max() - raw[normalize_cols].min())) * 20


st.write("## Model Predict")
model = load("heart_failure_predictor(1).joblib")
results = model.predict(normalized_df)
results = "**The model believes that the patient is under no immediate threat from heart failure**" if results[0] == 0 else "**_The patient is under risk of heart failure!_**"
st.write(results)