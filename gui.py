import streamlit as st
import tkinter as tk
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
water = pd.read_csv(r'water_potability.csv')
water['Sulfate'] = water['Sulfate'].fillna(water['Sulfate'].mean())
water['ph'] = water['ph'].fillna(water['ph'].mean())
water['Trihalomethanes'] = water['Trihalomethanes'].fillna(water['Trihalomethanes'].mean())

row_zero = water[water['Potability'] == 0]
row_one = water[water['Potability'] == 1]
from sklearn.utils import resample
from sklearn.utils import shuffle
water_minority_upsampled = resample(row_one, replace=True, n_samples=1998)  #this is called over sampling
water_balanced = pd.concat([row_zero, water_minority_upsampled])
Y = water_balanced['Potability']
X= water_balanced.drop(columns = ['Potability'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.set_page_config(
    page_title='Water Predictor',
    page_icon=r'icons8-water-48.png',
    initial_sidebar_state='collapsed' # Collapsed sidebar
)
def get(path:str): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    with open(path,"r") as p:
        return json.load(p)

def load_model(selected_model):
    if selected_model == "Logistic Regression":
        return joblib.load(open(r"log",'rb'))
    elif selected_model == "Decision Tree":
        return joblib.load(open(r"dc",'rb'))
    elif selected_model == "SVM":
        return joblib.load(open(r"svm",'rb'))

models = [ "Logistic Regression", "Decision Tree", "SVM"]
selected_model = st.sidebar.selectbox("Select a Model:", models)

# Load the selected model
model = load_model(selected_model)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)

def predict(ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity):
     features=np.array([ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]).reshape(1,-1)
     features= scaler.transform(features)
     predection=model.predict(features)
     return predection





with st.sidebar:
     
  choose = option_menu(None, ["Home" , "Graphs" , "About" , "Contact"],
                                             icons = ['house' , 'kanban','book' ,'person lines fill'],
                                             menu_icon="app-indicator", default_index=0,
                                             styles = {
                                    "container" :{"padding":"5!important " , "background-color":"#fafafa"},


                                     "icon":{"color":"#E0E0E0EF" , "font-size" :"25px"},


                                     "nav-link":{"font-size":"16px" , "text-align" : "left" ,"margin":"0px" , "--hover-color " :"#eee" },

                                     "nav-link-selected ":{"background-color":"#02ab21"},
                        }

                                             )
  
if choose=='Home':
      st.write('# Water Predictor')
      st.write('-----------------')
      st.subheader('Enter The Details To Get The Potability')
      ph=st.number_input("Enter the PH :",max_value=14,min_value=0)
      Hardness=st.number_input("Enter the Hardness :")
      Solids=st.number_input("Enter the Solids :")
      Chloramines=st.number_input("Enter the Chloramines amount :")
      Sulfate=st.number_input("Enter the Sulfate amount :")
      Conductivity=st.number_input("Enter the Conductivity :")
      Organic_carbon=st.number_input("Enter the Organic_carbon amount :")
      Trihalomethanes=st.number_input("Enter the Trihalomethanes amount :")
      Turbidity=st.number_input("Enter the Turbidity scale :")
      sample=predict(ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity)
      if st.button("Get the Result"):
          if sample==0:
              st.write("This Water is not Suitable for Drinking")
              st.image(r"360_F_85332688_wUZjugb65K0Qj2lDqyeVgk60tQMf5xIr.jpg")
              st.write("with accuracy ")
              st.write(acc)
              fig=sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
              plt.xlabel("Predicted label")
              plt.ylabel("True label")
              plt.title("Confusion Matrix")
              st.pyplot()
              report=classification_report(y_test, y_pred)
              st.text_area('Classification Report:', report, height=400)

        
              
             
              
               
          elif sample==1:
               st.write("This Water is Suitable for Drinking")
               st.image(r"download.jpeg")
               st.write("with accuracy ")
               st.write(acc)
               fig=sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
               plt.xlabel("Predicted label")
               plt.ylabel("True label")
               plt.title("Confusion Matrix")
               st.pyplot()
               report=classification_report(y_test, y_pred)
               st.text_area('Classification Report:', report, height=400)

            


               
           
   





elif choose=='Graphs':
      st.write("Relations")
      st.write("---------")
      st.image(r"REL.png")
      st.write("BOX PLOTS")
      st.write("---------")
      st.image(r"output.png")

      


elif choose=='About':
      st.write("About Us")
      st.write("Our water prediction model utilizes advanced algorithms to assess water quality and predict its potability.By analyzing various parameters such as pH levels, turbidity and hardness ,our model provides accurate assessments. With a focus on public health and environmental sustainability, our predictions empower communities and organizations to make informed decisions regarding water management and safety. Join us in our mission to ensure access to clean and safe drinking water for all.")






elif choose=='Contact':
      st.write("Contact Us")
      with st.form(key='columns_in_form2',clear_on_submit=True):
          st.write('##Please Help us to improve')
          name=st.text_input(label='Enter your Name')
          email=st.text_input(label='Enter your Email')
          message=st.text_input(label='Enter your Message')
          submitted=st.form_submit_button('Submit')
          if submitted:
              st.write('Thanks for your feedback .We are committed to improving .Stay tuned for updates.')
      


