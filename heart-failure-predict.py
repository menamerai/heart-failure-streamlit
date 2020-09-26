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


st.write("## Input Dataframe")
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


st.write("## *Model Predict*")
model = load("heart_failure_predictor(1).joblib")
results = model.predict(normalized_df)
results = "**_The model believes that the patient is under no immediate threat from heart failure._** Please, however, do note that the False Negative percentage of the model is quite pronounced, so more supervization is probably needed." if results[0] == 0 else "**_The patient is under risk of heart failure!_**"
st.write(results)


st.write("""
# Documentation
Some of the data customization options I gave have quite the medical background needed,
which is quite undesireable as it makes the usage of this webapp more confusing.
Thus, I will now write some slight background information regarding the data presented.

\n

* **Anaemia**: Basically the decrease of red blood cells, or hemoglobin. 
In a healthy person's blood there are more than just the red-looking substance as there are also plasma and plateles, 
so a decrease in hemoglobin might suggest a blockache, or in general how little oxygen is being carried around.

* **Diabetes**: This condition is probably famous, 
basically there's too much sugar in your blood. 
I don't know the relevance of the type of diabetes, 
since some are innate and genetic, while some are developed later in life. 
I'll assume this only concerns whether a person's blood glucose is high.

* **High blood pressure**: Exactly what it sounds like, your blood pressure is too high. 
Usually prevalent in older people with conditions like the aforementioned diabetes, 
or just an unsual sodium-rich diet.

* **Creatinine phosphokinase** (mcg/L): The level of an enzyme called CPK in the bloodstream. 
These are not typically abundant in the bloodstream, and leak out when tissues are damaged.

* **Ejection fraction** (%): The percentage of blood that leaves the heart in each contractions.

* **Platelets** (kiloplateletes/mL): The amount of plateles in the blood.

* **Serum creatinine** (mg/dL): Level of serum cretinine in the blood. 
All I know is that this thing is a waste product caused by the natural wear and tear of the body.

* **Serum sodium** (mEq/L): The amount of sodium, or... kinda salt presented in the body.
""")