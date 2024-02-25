import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
card=pd.read_csv("creditcard.csv")
# separatinglegitimate and fraudulent transactions
legitimate=card[card['Class']==0]
fraud=card[card['Class']==1]
# undersampling  legitimate transactions to balance the classes
legitimate_sample=legitimate.sample(n=len(fraud))
credit_card=pd.concat([legitimate_sample,fraud],axis=0)
# spliting  data into training and testing sets
x= credit_card.drop('Class',axis=1)
y=credit_card['Class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# training RandomForest model
model1=RandomForestClassifier()
model1.fit(x_train,y_train)
# evaluating model performance
train_acc= accuracy_score(model1.predict(x_train),y_train)
test_acc= accuracy_score(model1.predict(x_test),y_test)
# creating Streamlit app
st.title("Credit Card Fraud Detection")
input_df=st.text_input("Enter the Features")
input_df_split=input_df.split(',')
submit=st.button("Submit")
if submit:
    features=np.asarray(input_df_split,dtype=np.float64)
    prediction=model1.predict(features.reshape(1,-1))
    if prediction[0]==0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fraudulent Transaction")