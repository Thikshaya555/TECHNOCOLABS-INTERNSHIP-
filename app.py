import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image


pickle_in=open("lr_model.pkl","rb")
lr_model=pickle.load(pickle_in)


def predict_note_authentication(PAY_1,LIMIT_BAL):
	list1=[[PAY_1,LIMIT_BAL]]
	b=pd.DataFrame(list1,columns=['PAY_1','LIMIT_BAL'])
	prediction=lr_model.predict(b)[0]
	if prediction == 0:
		prediction1="account will not  default the next month"
	else:
		prediction1="account will default the next month" 
	print(prediction)
	return prediction1

def main():
	st.title("CREDIT CARD WORTHINESS")
	html_temp="""
	<div style="background-color:tomato;padding:10px">
	<h2 style="color:white;text-align:center;">CREDIT CARD WORTHINESS APP USING LOGISTIC REGRESSION</h2>
	</div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)
	PAY_1=st.text_input("PAY_1","Type Here")
	LIMIT_BAL=st.text_input("LIMIT_BAL","Type Here")
	result=""
	if st.button("Predict"):
		result=predict_note_authentication(PAY_1,LIMIT_BAL)
	st.success("The output is {}".format(result))
	if st.button("About"):
		st.text("LETS LEARN")
		st.text("BUILT WITH STREAMLIT")












if __name__=='__main__':
	main()
