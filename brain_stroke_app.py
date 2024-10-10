# import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib

import plotly.express as px
from PIL import Image


# Orginal excel data
df = pd.read_excel(r"D:\\ajay\\python\\brainstroke.xlsx")

# preprocessed data
df2 = pd.read_csv("D:\\ajay\\python\\preprocess_brain_stroke.csv")

# set web name
st.set_page_config(page_title = "stroke predict app")

# set the header name
st.header("Brain Stroke Prediction")

# write the introduction
st.write("""The Brain Stroke Prediction App built with Streamlit is an interactive tool designed to help users 
         assess their risk of experiencing a stroke. Utilizing a user-friendly interface, the app allows 
         individuals to input relevant health data, such as age, blood pressure, cholesterol levels, and 
         lifestyle habits. It employs machine learning algorithms to analyze this information and provide 
         personalized risk assessments. """)

# show the image
img = Image.open("D:\\ajay\\python\\brainstroke.jpeg")
st.image(img)


data = st.checkbox("Show Orginal Data")
#show the orginal data
if data:
    st.dataframe(df)

data_2 = st.checkbox("Show Preprocessed Data")
# show the preprocessed data
if data_2:
    st.dataframe(df2)

# show bar plot using matplotlib

plot_check = st.checkbox("If You Want Create Bar Plot")

if plot_check:
    col = list(df.columns)

    hu = st.selectbox("Select only stroke",col)

    x_var = st.selectbox("Select the column for Count plot",col)

    ax = px.bar(df , x= x_var,color = hu,barmode = 'group',color_discrete_sequence = ['red','green'])

    show = st.button("Submit")

    if show:
        st.plotly_chart(ax)


glucose_df = df.groupby(by = 'stroke',as_index = False)['avg_glucose_level'].mean()

sub_check = st.checkbox("If You Want See The All Plots")

if sub_check:

    st.success("Overall Analyse")
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 14))

    # axis 0,0 plot
    sns.countplot(x = df['stroke'],color = 'green',edgecolor = 'black',ax = axs[0][0])
    axs[0][0].set_title("Stroke")
    axs[0][0].set_xlabel("Stroke")
    axs[0][0].set_ylabel("Frequency")


    # axis 0 , 1 plot
    sns.countplot(x = df['gender'],hue = df['stroke'],edgecolor = 'black',ax = axs[0][1])
    axs[0][1].set_title("Gender Wise Stroke")
    axs[0][1].set_xlabel("Gender")
    axs[0][1].set_ylabel("Frequency")

    # axis 1, 0 plot
    sns.countplot(x = df['heart_disease'],hue = df['stroke'],palette= 'coolwarm',edgecolor = 'black',ax = axs[1][0])
    axs[1][0].set_title("Heart Disease Wise Stroke")
    axs[1][0].set_xlabel("Heart Disease")
    axs[1][0].set_ylabel("Count")

    # axis 1, 1 plot
    sns.countplot(x = df['work_type'],hue = df['stroke'],palette = 'RdYlBu',edgecolor = 'black',ax = axs[1][1])
    axs[1][1].set_title("Work Type Wise Stroke")
    axs[1][1].set_xlabel("Work Type")

    # axis 2,0 plot
    sns.countplot(x = df['Residence_type'],hue = df['stroke'],edgecolor = 'black',palette = 'deep',ax = axs[2][0])
    axs[2][0].set_xlabel("Residence Type")
    axs[2][0].set_ylabel("Count")
    axs[2][0].set_title("Area Wise Stroke")

    # axis 2,1 plot
    sns.barplot(x = 'stroke',y = 'avg_glucose_level',data = glucose_df,color = 'red',edgecolor = 'black',ax = axs[2][1])
    axs[2][1].set_xlabel("Stroke")
    axs[2][1].set_ylabel("Glucose Level")
    axs[2][1].set_title("Average Glucose Level of Stroke")

    st.pyplot(fig)


# give the input to prediction

# gender column
st.divider()
gend = df['gender'].unique()
gender = st.selectbox("Select Gender : ",gend)
st.success(f"You Selected {gender}")
gen = None

if gender == "Male":
    gen = 1
else:
    gen = 0

# age column
st.divider()
age = st.number_input(label = "Enter Your Age:",min_value = 0,max_value = 100)
st.success(f"You Entered {age}")


#hypertension column
st.divider()
hyper = df['hypertension'].unique()
hyperten = st.selectbox("If you have hypertension select Yes or No",hyper)
st.success(f"You selected {hyperten}")
tension = None

if hyperten == "Yes":
    tension = 1
else:
    tension = 0

# heart disease column
st.divider()
heart = df['heart_disease'].unique()
heart_disease = st.selectbox("If you have a heart disease select Yes or No : ",heart)
st.success(f"You selected {heart_disease}")
disease = None

if heart_disease == "Yes":
    disease = 1
else:
    disease = 0

# ever married column 
st.divider()
marry = df['ever_married'].unique()
ever_married = st.selectbox("If you married person select Yes : ",marry)
st.success(f"You Selected {ever_married}")
ever = None

if ever_married == "Yes":
    ever = 1
else:
    ever = 0

# work type column
st.divider()
work = df['work_type'].unique()
work_type = st.selectbox("Select Your Work Position : ",work)
st.success(f"You Selected {work_type}")
wtype = None

if work_type == "children":
    wtype = 0
elif work_type == "Govt_job":
    wtype = 1
elif work_type == "Self-employed":
    wtype = 2
else:
    wtype = 3

#residence type 
st.divider()
residence = df['Residence_type'].unique()
resi_type = st.selectbox("Select Your Residence Type : ",residence)
st.success(f"You Selected {resi_type}")
residence_type = None

if resi_type == 'Rural':
    residence_type = 1
else:
    residence_type = 0

# Glucose Level 
st.divider()
glucose = st.number_input(label = "Enter Your Average Glucose Level",min_value = 0.0,format = "%.2f")
st.success(f"You Entered {glucose}")

# bmi level column
st.divider()
bmi_level = st.number_input("Enter Your BMI Level : ",min_value = 0.0,format = "%.2f")
st.success(f"You Entered {bmi_level}")

# smoking status 
st.divider()
smoke = df['smoking_status'].unique()
smoke_sta = st.selectbox("Select Smoking Status : ",smoke)
st.warning("Unknown Values means Abscond Person Values")

status = None
if smoke_sta == "never smoked":
    status = 0
elif smoke_sta == "formerly smoked":
    status = 1
elif smoke_sta == "smokes":
    status = 2
else:
    status = 3

x_values = [gen,age,tension,disease,ever,wtype,residence_type,glucose,bmi_level,status]

orginal_value = [gender,age,hyperten,heart_disease,ever_married,work_type,resi_type,glucose,bmi_level,smoke_sta]

st.divider()
# predict the x value
model = joblib.load("D:\\ajay\\python\\brain_stroke_model.pkl")

predictbutton = st.button("Predict")

if predictbutton:
    x = np.array(x_values).reshape(1,-1)
    org_value = np.array(orginal_value).reshape(1,-1)
    org_value 
    prediction = model.predict(x)

    if prediction == 0:
        st.success("You Dont have a Brain Stroke")
    else:
        st.error("You have Brain stroke")


else:
    st.write("Please use the predict button")

